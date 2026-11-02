import math
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.seed_utils import set_all_seeds, get_rng_state_dict, set_rng_state_dict
from common.logging_utils import JsonlLogger, save_json, get_gpu_memory_mb, reset_peak_gpu_memory
from task1.data_prep import collate_sft_batch
from task1.adapters import export_adapter_state

RECOVERY_UPDATES = (312, 625, 937, 1250)

def build_adamw_optimizer(
    model: nn.Module,
    peak_lr: float,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build AdamW with weight_decay=0.01, excluding biases and parameters with < 2 dimensions."""
    decay_params, no_decay_params = [], []
    seen_ptrs = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or param.data_ptr() in seen_ptrs:
            continue
        seen_ptrs.add(param.data_ptr())
        if param.ndim < 2 or name.endswith(".bias") or "bias" in name.split(".")[-1]:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=peak_lr, betas=betas, eps=eps)

def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_updates: int = 38,
    total_updates: int = 1250,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for 38 updates, then cosine decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_updates:
            return float(current_step + 1) / float(max(1, warmup_updates))
        progress = float(current_step - warmup_updates) / float(max(1, total_updates - warmup_updates))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def snapshot_starting_parameters(
    model: nn.Module, ref_device: torch.device | str
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Snapshot unique starting parameters theta_0 for L2-SP and displacement calculation.
    Counts tied parameters once via data_ptr(). Tracks reference storage and transfer overhead.
    """
    t0 = time.perf_counter()
    seen_ptrs = set()
    theta0_list = []
    total_bytes = 0
    for param in model.parameters():
        if param.data_ptr() in seen_ptrs:
            continue
        seen_ptrs.add(param.data_ptr())
        ref_t = param.detach().clone().to(device=ref_device, dtype=torch.float32)
        theta0_list.append(ref_t)
        total_bytes += ref_t.numel() * 4
    transfer_sec = time.perf_counter() - t0
    return theta0_list, {
        "ref_storage_mb": total_bytes / (1024 ** 2),
        "ref_transfer_time_sec": transfer_sec,
        "unique_parameter_tensors": len(theta0_list),
    }

def compute_l2_sp_penalty(model: nn.Module, theta0_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute Omega(theta, theta_0) = 0.5 * sum_j (theta_j - theta_{0,j})^2 over all unique parameters,
    including biases and normalization parameters, counting tied parameters once. Does NOT average.
    """
    seen_ptrs = set()
    unique_params = []
    for param in model.parameters():
        if param.data_ptr() in seen_ptrs:
            continue
        seen_ptrs.add(param.data_ptr())
        unique_params.append(param)

    omega = torch.zeros((), device=unique_params[0].device, dtype=torch.float32)
    for p, p0 in zip(unique_params, theta0_list):
        diff = p.float() - p0.to(p.device, non_blocking=True)
        omega = omega + 0.5 * torch.sum(diff * diff)
    return omega

@torch.no_grad()
def compute_model_displacement_l2(model: nn.Module, theta0_list: List[torch.Tensor]) -> float:
    """Compute full-model L2 displacement ||theta - theta_0||_2 = sqrt(2 * Omega(theta, theta_0))."""
    omega = compute_l2_sp_penalty(model, theta0_list)
    return float(torch.sqrt(2.0 * omega).item())

def compute_microbatch_token_sum_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute summed next-token cross-entropy over valid assistant-response + EOT tokens."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )

def train_task1_variant(
    model: nn.Module,
    examples: List[Dict[str, Any]],
    variant_name: str,
    output_dir: str | Path,
    peak_lr: float,
    l2_sp_lambda: float = 0.0,
    microbatch_size: int = 2,
    grad_accum_steps: int = 16,
    total_updates: int = 1250,
    warmup_updates: int = 38,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    pad_token_id: int = 0,
    seed: int = 6304,
    is_adapter_run: bool = False,
    track_displacement: bool = False,
    recovery_updates: Tuple[int, ...] = RECOVERY_UPDATES,
    save_full_eval_checkpoints: Tuple[int, ...] = (),
    extra_overhead_info: Optional[Dict[str, Any]] = None,
    resume_from: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Unified Task 1 training loop for Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, and OPLoRA-16."""
    set_all_seeds(seed)
    out_dir = Path(output_dir) / variant_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "train_log.jsonl")

    device = next(model.parameters()).device
    use_cuda_amp = device.type == "cuda"

    # Enable gradient checkpointing as required
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if is_adapter_run and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    theta0_list: List[torch.Tensor] = []
    ref_overhead = {"ref_storage_mb": 0.0, "ref_transfer_time_sec": 0.0}
    if l2_sp_lambda > 0.0 or track_displacement:
        # Keep theta_0 on GPU if L2-SP is active every step; CPU if only used for midpoint/final displacement
        ref_dev = device if l2_sp_lambda > 0.0 else torch.device("cpu")
        theta0_list, ref_overhead = snapshot_starting_parameters(model, ref_device=ref_dev)

    optimizer = build_adamw_optimizer(model, peak_lr=peak_lr, weight_decay=weight_decay)
    scheduler = build_cosine_warmup_scheduler(optimizer, warmup_updates=warmup_updates, total_updates=total_updates)
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    start_update = 0
    skipped_updates = 0
    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        set_rng_state_dict(ckpt["rng_state"])
        start_update = int(ckpt["update"])
        skipped_updates = int(ckpt.get("skipped_updates", 0))

    effective_batch_size = microbatch_size * grad_accum_steps
    reset_peak_gpu_memory()
    model.train()
    t_start = time.perf_counter()

    displacement_records = {}
    checkpoint_sizes_mb = {}
    history = []

    for update_idx in range(start_update, total_updates):
        update_num = update_idx + 1
        batch_start = update_idx * effective_batch_size
        batch_end = min(len(examples), batch_start + effective_batch_size)
        effective_slice = examples[batch_start:batch_end]

        total_valid_tokens = max(1, sum(ex["num_valid_tokens"] for ex in effective_slice))
        optimizer.zero_grad(set_to_none=True)
        sft_loss_accum = 0.0

        for m_idx in
