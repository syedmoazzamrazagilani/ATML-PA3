import gc
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from common.seed_utils import set_all_seeds, get_rng_state_dict, set_rng_state_dict
from common.logging_utils import JsonlLogger, save_json, get_gpu_memory_mb, reset_peak_gpu_memory
from task2.evaluate_vlm import evaluate_development_suites

STAGE_CONFIGS = {
    1: {
        "name": "stage1_caption_alignment",
        "start_update": 0,
        "end_update": 2250,
        "num_updates": 2250,
        "connector_lr": 1e-3,
        "lora_lr": 0.0,
        "enable_lora": False,
    },
    2: {
        "name": "stage2_visual_instruction_tuning",
        "start_update": 2250,
        "end_update": 3750,
        "num_updates": 1500,
        "connector_lr": 1e-4,
        "lora_lr": 2e-5,
        "enable_lora": True,
        "draw_probs": {"presence": 0.50, "counting": 0.25, "caption": 0.25},
    },
    3: {
        "name": "stage3_natural_vqa_specialization",
        "start_update": 3750,
        "end_update": 4500,
        "num_updates": 750,
        "connector_lr": 5e-5,
        "lora_lr": 1e-5,
        "enable_lora": True,
        "draw_probs": {
            "natural_vqa": 0.70,
            "presence": 0.10,
            "counting": 0.10,
            "caption": 0.10,
        },
    },
}

DEV_EVAL_UPDATES = (0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500)


def build_vlm_stage_optimizer(
    vlm_model: nn.Module,
    connector_lr: float,
    lora_lr: float = 0.0,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Build AdamW with separate peak learning rates for connector and LoRA adapters.
    Applies weight_decay=0.01 only to >=2D non-bias parameters; uses foreach=False to save VRAM.
    """
    conn_decay, conn_nodecay = [], []
    for name, p in vlm_model.connector.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "bias" in name:
            conn_nodecay.append(p)
        else:
            conn_decay.append(p)

    param_groups = [
        {"params": conn_decay, "lr": connector_lr, "weight_decay": weight_decay, "group_name": "connector_decay"},
        {"params": conn_nodecay, "lr": connector_lr, "weight_decay": 0.0, "group_name": "connector_nodecay"},
    ]

    if lora_lr > 0.0:
        lora_decay, lora_nodecay = [], []
        for name, p in vlm_model.qwen.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or name.endswith(".bias") or "bias" in name:
                lora_nodecay.append(p)
            else:
                lora_decay.append(p)
        if lora_decay:
            param_groups.append(
                {"params": lora_decay, "lr": lora_lr, "weight_decay": weight_decay, "group_name": "lora_decay"}
            )
        if lora_nodecay:
            param_groups.append(
                {"params": lora_nodecay, "lr": lora_lr, "weight_decay": 0.0, "group_name": "lora_nodecay"}
            )

    return torch.optim.AdamW(param_groups, betas=betas, eps=eps, foreach=False)


def build_stage_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    stage_updates: int,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.10,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-stage 5% linear warmup, then cosine decay to 10% of peak LR."""
    warmup_steps = max(1, int(round(stage_updates * warmup_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, stage_updates - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class CurriculumBatchSampler:
    """
    Deterministic seeded sampler for the 3-stage VLM curriculum.
    - Stage 1: two passes over the 18,000 caption examples (2,250 updates x 16 examples).
    - Stage 2: each draw contributes 2 examples (1 question pair or 2 caption examples)
      with probabilities 50% presence, 25% counting, 25% caption replay (8 draws per 16-example batch).
    - Stage 3: each draw contributes 2 examples with probabilities
      70% natural VQAv2, 10% presence, 10% counting, 10% caption replay (8 draws per 16-example batch).
    """
    def __init__(
        self,
        caption_examples: List[Dict[str, Any]],
        presence_pairs: List[Dict[str, Any]],
        counting_pairs: List[Dict[str, Any]],
        natural_vqa_pairs: List[Dict[str, Any]],
        seed: int = 6304,
    ):
        self.caption_examples = caption_examples
        self.presence_pairs = presence_pairs
        self.counting_pairs = counting_pairs
        self.natural_vqa_pairs = natural_vqa_pairs
        self.rng = np.random.default_rng(seed)

        # Pre-build two deterministic passes of the 18,000 caption examples for Stage 1
        n_cap = len(caption_examples)
        pass1 = self.rng.permutation(n_cap).tolist()
        pass2 = self.rng.permutation(n_cap).tolist()
        self.stage1_order = pass1 + pass2
        self.sampler_position = 0
        self.mixture_counts = {
            "stage1": {"caption": 0},
            "stage2": {"presence": 0, "counting": 0, "caption": 0},
            "stage3": {"natural_vqa": 0, "presence": 0, "counting": 0, "caption": 0},
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "sampler_position": self.sampler_position,
            "mixture_counts": self.mixture_counts,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng_state"]
        self.sampler_position = int(state["sampler_position"])
        self.mixture_counts = state["mixture_counts"]

    def _pair_to_two_examples(self, pair_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Unpack a complementary pair into its two (image, question, target_answer) examples."""
        q_text = pair_item["question"]
        return [
            {
                "source_type": pair_item.get("source_type", "vqa"),
                "pair_id": pair_item["pair_id"],
                "image_id": pair_item["image_a_id"],
                "question": q_text,
                "target_text": pair_item["answer_a"],
                "is_caption": False,
            },
            {
                "source_type": pair_item.get("source_type", "vqa"),
                "pair_id": pair_item["pair_id"],
                "image_id": pair_item["image_b_id"],
                "question": q_text,
                "target_text": pair_item["answer_b"],
                "is_caption": False,
            },
        ]

    def next_effective_batch(self, stage: int, effective_batch_size: int = 16) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        n_cap = len(self.caption_examples)

        if stage == 1:
            for _ in range(effective_batch_size):
                idx = self.stage1_order[self.sampler_position % len(self.stage1_order)]
                self.sampler_position += 1
                batch.append(self.caption_examples[idx])
                self.mixture_counts["stage1"]["caption"] += 1
            return batch

        num_draws = effective_batch_size // 2
        if stage == 2:
            components = ["presence", "counting", "caption"]
            probs = [0.50, 0.25, 0.25]
            stage_key = "stage2"
        else:
            components = ["natural_vqa", "presence", "counting", "caption"]
            probs = [0.70, 0.10, 0.10, 0.10]
            stage_key = "stage3"

        for _ in range(num_draws):
            comp = str(self.rng.choice(components, p=probs))
            self.mixture_counts[stage_key][comp] += 2
            self.sampler_position += 1
            if comp == "caption":
                c_idx = self.rng.integers(0, n_cap, size=2).tolist()
                batch.append(self.caption_examples[c_idx[0]])
                batch.append(self.caption_examples[c_idx[1]])
            elif comp == "presence":
                p_idx = int(self.rng.integers(0, len(self.presence_pairs)))
                batch.extend(self._pair_to_two_examples(self.presence_pairs[p_idx]))
            elif comp == "counting":
                c_idx = int(self.rng.integers(0, len(self.counting_pairs)))
                batch.extend(self._pair_to_two_examples(self.counting_pairs[c_idx]))
            elif comp == "natural_vqa":
                v_idx = int(self.rng.integers(0, len(self.natural_vqa_pairs)))
                batch.extend(self._pair_to_two_examples(self.natural_vqa_pairs[v_idx]))

        return batch


def train_vlm_curriculum(
    vlm_model: nn.Module,
    tokenizer: Any,
    feature_store: Any,
    train_pools: Dict[str, List[Dict[str, Any]]],
    dev_suites: Dict[str, List[Dict[str, Any]]],
    output_dir: str | Path,
    microbatch_size: int = 8,
    grad_accum_steps: int = 2,
    max_grad_norm: float = 1.0,
    seed: int = 6304,
    is_smoke: bool = False,
    resume_from: Optional[str | Path] = None,
    prune_recovery_checkpoints: bool = True,
) -> Dict[str, Any]:
    """
    Train the SigLIP-Qwen VLM sequentially through Stages 1, 2, and 3.
    Tracks training loss, mixture counts, dev caption loss, dev answer NLL (under correct,
    complementary, and no-image inputs), image-swap penalty Delta_NLL, trainable parameters,
    runtime, and peak GPU memory by stage.
    """
    set_all_seeds(seed)
    out_dir = Path(output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_logger = JsonlLogger(out_dir / "train_log.jsonl")
    dev_logger = JsonlLogger(out_dir / "dev_eval_log.jsonl")

    device = next(vlm_model.parameters()).device
    use_cuda_amp = device.type == "cuda"
    effective_batch_size = microbatch_size * grad_accum_steps

    sampler = CurriculumBatchSampler(
        caption_examples=train_pools["captions"],
        presence_pairs=train_pools["presence_pairs"],
        counting_pairs=train_pools["counting_pairs"],
        natural_vqa_pairs=train_pools["natural_vqa_pairs"],
        seed=seed,
    )

    # Scale down update budgets for fast smoke testing
    if is_smoke:
        stage_schedule = {
            1: {**STAGE_CONFIGS[1], "start_update": 0, "end_update": 2, "num_updates": 2},
            2: {**STAGE_CONFIGS[2], "start_update": 2, "end_update": 4, "num_updates": 2},
            3: {**STAGE_CONFIGS[3], "start_update": 4, "end_update": 6, "num_updates": 2},
        }
        dev_eval_set = {0, 2, 4, 6}
        ckpt_interval = 2
    else:
        stage_schedule = STAGE_CONFIGS
        dev_eval_set = set(DEV_EVAL_UPDATES)
        ckpt_interval = 250

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)

    global_update = 0
    skipped_updates = 0
    train_history: List[Dict[str, Any]] = []
    dev_history: List[Dict[str, Any]] = []
    stage_resources: Dict[str, Any] = {}

    if resume_from is not None and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device)
        global_update = int(ckpt["global_update"])
        skipped_updates = int(ckpt.get("skipped_updates", 0))
        sampler.load_state_dict(ckpt["sampler_state"])
        if ckpt.get("lora_enabled", False) and not vlm_model.lora_enabled:
            vlm_model.enable_stage2_lora(r=16, alpha=32.0, dropout=0.05, init_seed=seed)
        vlm_model.load_trainable_state_dict(ckpt["trainable_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        set_rng_state_dict(ckpt["rng_state"])

    # Pre-training Development Evaluation (Update 0)
    if global_update == 0 and 0 in dev_eval_set:
        dev0 = evaluate_development_suites(
            vlm_model, tokenizer, feature_store, dev_suites, update_num=0, stage_num=0, is_smoke=is_smoke
        )
        dev_logger.log(dev0)
        dev_history.append(dev0)

    for stage_num in (1, 2, 3):
        scfg = stage_schedule[stage_num]
        if global_update >= scfg["end_update"]:
            continue

        if scfg["enable_lora"] and not vlm_model.lora_enabled:
            vlm_model.enable_stage2_lora(r=16, alpha=32.0, dropout=0.05, init_seed=seed)

        optimizer = build_vlm_stage_optimizer(
            vlm_model,
            connector_lr=scfg["connector_lr"],
            lora_lr=scfg["lora_lr"],
            weight_decay=0.01,
        )
        scheduler = build_stage_cosine_scheduler(
            optimizer, stage_updates=scfg["num_updates"], warmup_ratio=0.05, min_lr_ratio=0.10
        )

        # If resuming mid-stage, advance scheduler to current step within stage
        stage_offset = max(0, global_update - scfg["start_update"])
        for _ in range(stage_offset):
            scheduler.step()

        reset_peak_gpu_memory()
        t_stage_start = time.perf_counter()
        vlm_model.train()

        for step_in_stage in range(stage_offset, scfg["num_updates"]):
            global_update += 1
            eff_batch_items = sampler.next_effective_batch(stage_num, effective_batch_size)

            optimizer.zero_grad(set_to_none=True)
            batch_loss_accum = 0.0

            for m_idx in range(grad_accum_steps):
                mb_items = eff_batch_items[
                    m_idx * microbatch_size : (m_idx + 1) * microbatch_size
                ]
                if not mb_items:
                    continue

                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_cuda_amp
                ):
                    # forward_example_weighted_loss averages token CE within each example,
                    # then sums across microbatch; dividing by effective_batch_size (16)
                    # yields exact example-weighted batch loss across B=16.
                    mb_loss_sum = vlm_model.forward_example_weighted_loss_sum(
                        mb_items, tokenizer, feature_store
                    )
                    mb_loss = mb_loss_sum / float(effective_batch_size)

                scaler.scale(mb_loss).backward()
                batch_loss_accum += float(mb_loss.detach().item())
                del mb_loss_sum, mb_loss

            scaler.unscale_(optimizer)
            trainable_params_list = [p for p in vlm_model.parameters() if p.requires_grad]
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable_params_list, max_grad_norm).item()
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

            conn_lr_now = float(optimizer.param_groups[0]["lr"])
            lora_lr_now = (
                float(optimizer.param_groups[2]["lr"])
                if len(optimizer.param_groups) > 2
                else 0.0
            )

            rec = {
                "update": global_update,
                "stage": stage_num,
                "examples_seen": global_update * effective_batch_size,
                "loss": batch_loss_accum,
                "connector_lr": conn_lr_now,
                "lora_lr": lora_lr_now,
                "grad_norm": grad_norm,
                "step_skipped": bool(step_skipped),
                "skipped_updates_total": skipped_updates,
            }
            train_logger.log(rec)
            train_history.append(rec)

            # Save resumable checkpoint every 250 updates
            if global_update % ckpt_interval == 0 or global_update == scfg["end_update"]:
                ckpt_path = ckpt_dir / f"resumable_update_{global_update}.pt"
                torch.save(
                    {
                        "global_update": global_update,
                        "stage": stage_num,
                        "lora_enabled": vlm_model.lora_enabled,
                        "trainable_state_dict": vlm_model.get_trainable_state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "sampler_state": sampler.state_dict(),
                        "rng_state": get_rng_state_dict(),
                        "skipped_updates": skipped_updates,
                    },
                    ckpt_path,
                )
                if prune_recovery_checkpoints:
                    prev_u = global_update - ckpt_interval
                    prev_ckpt = ckpt_dir / f"resumable_update_{prev_u}.pt"
                    if prev_ckpt.exists():
                        prev_ckpt.unlink()

            # Development Evaluation at scheduled updates (every 500 + stage endpoints)
            if global_update in dev_eval_set:
                dev_metrics = evaluate_development_suites(
                    vlm_model,
                    tokenizer,
                    feature_store,
                    dev_suites,
                    update_num=global_update,
                    stage_num=stage_num,
                    is_smoke=is_smoke,
                )
                dev_logger.log(dev_metrics)
                dev_history.append(dev_metrics)
                vlm_model.train()

        stage_time_sec = time.perf_counter() - t_stage_start
        mem_stats = get_gpu_memory_mb()
        param_counts = vlm_model.get_parameter_breakdown()

        # Save stage-endpoint connector/adapter weights (2250, 3750, 4500)
        stage_weights_path = ckpt_dir / f"stage_{stage_num}_update_{global_update}_weights.pt"
        torch.save(vlm_model.get_trainable_state_dict(), stage_weights_path)

        stage_resources[f"stage_{stage_num}"] = {
            "stage": stage_num,
            "name": scfg["name"],
            "end_update": global_update,
            "training_time_sec": stage_time_sec,
            "peak_gpu_memory_mb": mem_stats["max_allocated_mb"],
            "checkpoint_size_mb": stage_weights_path.stat().st_size / (1024 ** 2),
            **param_counts,
        }
        del optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "seed": seed,
        "visual_tokens": vlm_model.num_visual_tokens,
        "calibration_info": vlm_model.calibration_info,
        "mixture_counts": sampler.mixture_counts,
        "stage_resources": stage_resources,
        "skipped_updates": skipped_updates,
        "final_train_loss": train_history[-1]["loss"] if train_history else None,
    }
    save_json(summary, out_dir / "training_summary.json")
    return {
        "summary": summary,
        "train_history": train_history,
        "dev_history": dev_history,
    }
