import gc
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
    """
    Build AdamW with weight_decay=0.01, excluding biases and parameters with < 2 dimensions.
    Uses foreach=False to avoid multi-gigabyte temporary buffer allocations during optimizer.step().
    """
    decay_params = []
    no_decay_params = []
    seen_ptrs = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.data_ptr() in seen_ptrs:
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
    return torch.optim.AdamW(
        optim_groups, lr=peak_lr, betas=betas, eps=eps, foreach=False
    )


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_updates: int = 38,
    total_updates: int = 1250,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for `warmup_updates` steps, then cosine decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_updates:
            return float(current_step + 1) / float(max(1, warmup_updates))
        progress = float(current_step - warmup_updates) / float(
            max(1, total_updates - warmup_updates)
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def snapshot_starting_parameters(
    model: nn.Module, store_device: torch.device | str = "cpu"
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Snapshot unique starting parameters theta_0 for L2-SP and displacement calculation.
    Counts tied parameters once via data_ptr().
    Stores in FP16 (since pretrained Qwen3 weights are natively 16-bit) and upcasts to FP32 on the fly.
    """
    t0 = time.perf_counter()
    seen_ptrs = set()
    theta0_list = []
    total_bytes_fp32 = 0
    for param in model.parameters():
        if param.data_ptr() in seen_ptrs:
            continue
        seen_ptrs.add(param.data_ptr())
        ref_t = param.detach().to(device=store_device, dtype=torch.float16).clone()
        theta0_list.append(ref_t)
        total_bytes_fp32 += ref_t.numel() * 4
    transfer_sec = time.perf_counter() - t0
    return theta0_list, {
        "ref_storage_mb": total_bytes_fp32 / (1024 ** 2),
        "ref_transfer_time_sec": transfer_sec,
        "unique_parameter_tensors": len(theta0_list),
    }


@torch.no_grad()
def compute_and_apply_l2_sp(
    model: nn.Module,
    theta0_list: List[torch.Tensor],
    l2_sp_lambda: float = 0.0,
    grad_scale: float = 1.0,
) -> float:
    """
    Compute exact Omega(theta, theta_0) = 0.5 * sum_j (theta_j - theta_{0,j})^2 in FP32 over all
    unique parameters (counting tied parameters once, without averaging).
    When l2_sp_lambda > 0, adds the exact analytical gradient (grad_scale * lambda * (theta - theta_0))
    directly to param.grad tensor-by-tensor so zero autograd graph memory is consumed.
    """
    seen_ptrs = set()
    unique_params = []
    for param in model.parameters():
        if param.data_ptr() in seen_ptrs:
            continue
        seen_ptrs.add(param.data_ptr())
        unique_params.append(param)

    omega_sum = 0.0
    for p, p0 in zip(unique_params, theta0_list):
        p0_f32 = p0.to(device=p.device, dtype=torch.float32, non_blocking=True)
        diff = p.detach().to(torch.float32) - p0_f32
        omega_sum += 0.5 * float(torch.sum(diff * diff).item())
        if l2_sp_lambda > 0.0 and p.requires_grad:
            grad_term = (grad_scale * l2_sp_lambda * diff).to(dtype=p.dtype)
            if p.grad is None:
                p.grad = grad_term
            else:
                p.grad.add_(grad_term)
        del diff, p0_f32
    return omega_sum


@torch.no_grad()
def compute_model_displacement_l2(
    model: nn.Module, theta0_list: List[torch.Tensor]
) -> float:
    """Compute full-model L2 displacement ||theta - theta_0||_2 = sqrt(2 * Omega(theta, theta_0))."""
    omega = compute_and_apply_l2_sp(model, theta0_list, l2_sp_lambda=0.0, grad_scale=1.0)
    return float(math.sqrt(max(0.0, 2.0 * omega)))


def forward_and_compute_mb_loss_sum(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Compute exact summed next-token cross-entropy over valid assistant-response + EOT tokens.
    When model has `.model` and `.lm_head` (e.g., Qwen3ForCausalLM), projects only valid unmasked
    token positions through `lm_head` in chunks of `chunk_size` to avoid allocating multi-gigabyte
    full-sequence (B, S, 151936) FP32 logit buffers.
    """
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        backbone_out = model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        hidden_states = (
            backbone_out.last_hidden_state
            if hasattr(backbone_out, "last_hidden_state")
            else backbone_out[0]
        )
        shift_h = hidden_states[..., :-1, :]
        shift_labels = labels[..., 1:]
        valid_mask = shift_labels != -100
        valid_h = shift_h[valid_mask]
        valid_targets = shift_labels[valid_mask]

        if valid_targets.numel() == 0:
            return hidden_states.sum() * 0.0

        lm_head = model.lm_head

        def _chunk_ce(h_chunk: torch.Tensor, t_chunk: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
            logits_chunk = F.linear(h_chunk, weight, bias).float()
            return F.cross_entropy(logits_chunk, t_chunk, reduction="sum")

        total_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        n_valid = valid_targets.size(0)
        for start in range(0, n_valid, chunk_size):
            end = min(n_valid, start + chunk_size)
            h_c = valid_h[start:end]
            t_c = valid_targets[start:end]
            if hidden_states.requires_grad:
                loss_c = checkpoint(
                    _chunk_ce, h_c, t_c, lm_head.weight, lm_head.bias, use_reentrant=False
                )
            else:
                loss_c = _chunk_ce(h_c, t_c, lm_head.weight, lm_head.bias)
            total_loss = total_loss + loss_c
        return total_loss

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
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
    resume_from: Optional[str | Path] = None,
    adapter_meta: Optional[Dict[str, float]] = None,
    prune_optimizer_states: bool = False,
) -> Dict[str, Any]:
    """Unified Task 1 trainer supporting Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, and OPLoRA-16."""
    effective_batch_size = microbatch_size * grad_accum_steps
    if len(examples) < effective_batch_size * total_updates:
        raise ValueError(
            f"Insufficient examples ({len(examples)}) for {total_updates} updates "
            f"with effective_batch_size={effective_batch_size}!"
        )

    set_all_seeds(seed)
    out_dir = Path(output_dir) / variant_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir / "train_log.jsonl")

    device = next(model.parameters()).device
    use_cuda_amp = device.type == "cuda"

    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    theta0_list: List[torch.Tensor] = []
    ref_overhead = {"ref_storage_mb": 0.0, "ref_transfer_time_sec": 0.0}
    if l2_sp_lambda > 0.0 or track_displacement:
        # Store on GPU in FP16 if needed every step (L2-SP), or on CPU if only needed at 625 & 1250 (Full FT)
        store_dev = device if l2_sp_lambda > 0.0 else "cpu"
        theta0_list, ref_overhead = snapshot_starting_parameters(model, store_device=store_dev)

    optimizer = build_adamw_optimizer(model, peak_lr=peak_lr, weight_decay=weight_decay)
    scheduler = build_cosine_warmup_scheduler(
        optimizer, warmup_updates=warmup_updates, total_updates=total_updates
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    start_update = 0
    skipped_updates = 0
    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "rng_state" in ckpt:
            set_rng_state_dict(ckpt["rng_state"])
        start_update = int(ckpt["update"])
        skipped_updates = int(ckpt.get("skipped_updates", 0))

    reset_peak_gpu_memory()
    model.train()
    t_start = time.perf_counter()

    displacement_records = {}
    checkpoint_sizes_mb = {}
    export_times_sec = {}
    history = []
    mid_update = 1 if total_updates == 2 else 625

    for update_idx in range(start_update, total_updates):
        update_num = update_idx + 1
        batch_start = update_idx * effective_batch_size
        batch_end = min(len(examples), batch_start + effective_batch_size)
        effective_slice = examples[batch_start:batch_end]

        total_valid_tokens = max(
            1, sum(ex["num_valid_tokens"] for ex in effective_slice)
        )

        optimizer.zero_grad(set_to_none=True)
        sft_loss_accum = 0.0

        for m_idx in range(grad_accum_steps):
            mb_slice = effective_slice[
                m_idx * microbatch_size : (m_idx + 1) * microbatch_size
            ]
            if not mb_slice:
                continue
            mb = collate_sft_batch(mb_slice, pad_token_id=pad_token_id)
            input_ids = mb["input_ids"].to(device, non_blocking=True)
            labels = mb["labels"].to(device, non_blocking=True)
            attention_mask = mb["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_cuda_amp
            ):
                mb_loss_sum = forward_and_compute_mb_loss_sum(
                    model, input_ids, attention_mask, labels, chunk_size=128
                )
                mb_loss = mb_loss_sum / float(total_valid_tokens)

            scaler.scale(mb_loss).backward()
            sft_loss_accum += float(mb_loss.detach().item())
            del mb_loss_sum, mb_loss, input_ids, labels, attention_mask

        omega_val = 0.0
        reg_val = 0.0
        if l2_sp_lambda > 0.0:
            cur_scale = float(scaler.get_scale()) if use_cuda_amp else 1.0
            omega_val = compute_and_apply_l2_sp(
                model, theta0_list, l2_sp_lambda=l2_sp_lambda, grad_scale=cur_scale
            )
            reg_val = l2_sp_lambda * omega_val

        scaler.unscale_(optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm
            ).item()
        )

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        step_skipped = use_cuda_amp and (scale_after < scale_before)
        if step_skipped:
            skipped_updates += 1
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        current_lr = float(optimizer.param_groups[0]["lr"])
        total_loss_val = sft_loss_accum + reg_val

        step_record = {
            "update": update_num,
            "examples_seen": batch_end,
            "sft_loss": sft_loss_accum,
            "omega": omega_val,
            "lambda_omega": reg_val,
            "total_loss": total_loss_val,
            "lr": current_lr,
            "grad_norm": grad_norm,
            "step_skipped": bool(step_skipped),
            "skipped_updates_total": skipped_updates,
            "valid_tokens_in_batch": total_valid_tokens,
        }

        if track_displacement and (update_num in (mid_update, total_updates)):
            disp = compute_model_displacement_l2(model, theta0_list)
            step_record["displacement_l2"] = disp
            displacement_records[f"update_{update_num}"] = disp

        if update_num in recovery_updates or update_num == total_updates:
            ckpt_path = ckpt_dir / f"checkpoint_update_{update_num}.pt"
            if is_adapter_run:
                export_info = export_adapter_state(
                    model,
                    ckpt_dir / f"adapter_update_{update_num}.pt",
                    export_projected=True,
                )
                step_record.update(export_info)
                checkpoint_sizes_mb[f"adapter_update_{update_num}"] = export_info["checkpoint_size_mb"]
                export_times_sec[f"adapter_update_{update_num}"] = export_info["export_time_sec"]

            trainable_state = {
                k: v.detach().cpu()
                for k, v in model.named_parameters()
                if v.requires_grad
            }
            full_recovery_payload = {
                "update": update_num,
                "examples_seen": batch_end,
                "data_position": batch_end,
                "model_state_dict": trainable_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": get_rng_state_dict(),
                "skipped_updates": skipped_updates,
            }
            torch.save(full_recovery_payload, ckpt_path)
            checkpoint_sizes_mb[f"recovery_update_{update_num}"] = (
                ckpt_path.stat().st_size / (1024 ** 2)
            )
            del full_recovery_payload

            # If prune_optimizer_states is enabled (to fit Kaggle's 19.5 GB disk),
            # keep only evaluation weights in FP16 for needed checkpoints and remove temporary recovery files
            if prune_optimizer_states and not is_adapter_run:
                keep_for_eval = (
                    (variant_name == "full_ft" and update_num in (mid_update, total_updates))
                    or (variant_name == "l2_sp" and update_num == total_updates)
                )
                if keep_for_eval:
                    fp16_state = {k: v.to(torch.float16) for k, v in trainable_state.items()}
                    torch.save(
                        {"update": update_num, "examples_seen": batch_end, "model_state_dict": fp16_state},
                        ckpt_path,
                    )
                    del fp16_state
                else:
                    ckpt_path.unlink(missing_ok=True)
            del trainable_state

        logger.log(step_record)
        history.append(step_record)

    elapsed_sec = time.perf_counter() - t_start
    mem_stats = get_gpu_memory_mb()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    del optimizer, scheduler, scaler, theta0_list
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "variant": variant_name,
        "trainable_params": trainable_params,
        "training_time_sec": elapsed_sec,
        "peak_gpu_memory_mb": mem_stats["max_allocated_mb"],
        "skipped_updates": skipped_updates,
        "displacement_l2": displacement_records,
        "reference_overhead": ref_overhead,
        "adapter_meta": adapter_meta or {},
        "checkpoint_sizes_mb": checkpoint_sizes_mb,
        "export_times_sec": export_times_sec,
        "final_sft_loss": history[-1]["sft_loss"] if history else None,
        "final_total_loss": history[-1]["total_loss"] if history else None,
    }
    save_json(summary, out_dir / "run_summary.json")
    return {"summary": summary, "history": history}
