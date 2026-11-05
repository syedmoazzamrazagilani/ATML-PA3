import gc
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import torch
import torch.nn as nn

from common.seed_utils import set_all_seeds, get_rng_state_dict
from common.logging_utils import JsonlLogger, save_json, get_gpu_memory_mb, reset_peak_gpu_memory
from task1.adapters import LoRALinear
from task2.model_vlm import SiglipQwenVLM

DEV_EVAL_STEPS = (0, 500, 1000, 1500, 2000, 2250, 2500, 3000, 3500, 3750, 4000, 4500)
CAPTION_PROMPT = "Describe this image."


def build_stage_optimizer_and_scheduler(
    vlm: SiglipQwenVLM,
    connector_peak_lr: float,
    lora_peak_lr: Optional[float],
    stage_updates: int,
    weight_decay: float = 0.01,
    min_lr_ratio: float = 0.10,
) -> Tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """
    Build AdamW with separate connector and LoRA parameter groups, weight_decay=0.01
    (excluding biases and <2D tensors), per-stage 5% warmup, and cosine decay to 10% of peak LR.
    """
    conn_decay, conn_nodecay = [], []
    for name, p in vlm.connector.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias"):
            conn_nodecay.append(p)
        else:
            conn_decay.append(p)

    param_groups = [
        {"params": conn_decay, "lr": connector_peak_lr, "weight_decay": weight_decay, "group_name": "connector_decay"},
        {"params": conn_nodecay, "lr": connector_peak_lr, "weight_decay": 0.0, "group_name": "connector_nodecay"},
    ]

    if lora_peak_lr is not None and lora_peak_lr > 0.0:
        lora_decay = [
            p for p in vlm.qwen.parameters() if p.requires_grad and p.ndim >= 2
        ]
        if lora_decay:
            param_groups.append(
                {
                    "params": lora_decay,
                    "lr": lora_peak_lr,
                    "weight_decay": weight_decay,
                    "group_name": "lora_decay",
                }
            )

    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.999), eps=1e
