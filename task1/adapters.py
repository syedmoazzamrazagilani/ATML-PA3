import math
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

TASK1_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

@torch.no_grad()
def compute_top_k_svd_bases(weight: torch.Tensor, k: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute leading k left and right singular vector matrices U_k (d_out x k), V_k (d_in x k) in FP32."""
    W_fp32 = weight.detach().to(torch.float32)
    U, _, Vh = torch.linalg.svd(W_fp32, full_matrices=False)
    k_eff = min(k, U.shape[1], Vh.shape[0])
    U_k = U[:, :k_eff].contiguous()
    V_k = Vh[:k_eff, :].transpose(0, 1).contiguous()
    return U_k, V_k

class LoRALinear(nn.Module):
    """
    LoRA and OPLoRA wrapper for a frozen Linear layer.
    Backbone weight W_0 remains frozen (FP16 on GPU), while A (r x d_in) and B (d_out x r) are FP32.
    For OPLoRA, leading singular bases U_k (d_out x k) and V_k (d_in x k) are frozen in FP32:
        B_tilde = B - U_k (U_k^T B),   A_tilde = A - (A V_k) V_k^T
        Delta W = (alpha / r) B_tilde A_tilde
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
        use_oplora: bool = False,
        protection_rank_k: int = 128,
        precomputed_bases: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        init_seed: int = 6304,
        layer_tag: str = "",
    ):
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)
        self.use_oplora = use_oplora
        self.protection_rank_k = protection_rank_k
        self.layer_tag = layer_tag

        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Deterministic per-layer initialization so LoRA-16 and OPLoRA-16 start with identical (A, B)
        gen = torch.Generator(device="cpu")
        tag_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(layer_tag))
        gen.manual_seed(init_seed + tag_hash)

        A_init = torch.empty((r, self.in_features), dtype=torch.float32)
        nn.init.kaiming_uniform_(A_init, a=math.sqrt(5), generator=gen)
        B_init = torch.zeros((self.out_features, r), dtype=torch.float32)

        device = base_layer.weight.device
        self.lora_A = nn.Parameter(A_init.to(device))
        self.lora_B = nn.Parameter(B_init.to(device))

        if self.use_oplora:
            if precomputed_bases is not None:
                U_k, V_k = precomputed_bases
            else:
                U_k, V_k = compute_top_k_svd_bases(self.base_layer.weight.data, k=protection_rank_k)
            self.register_buffer("U_k", U_k.to(device=device, dtype=torch.float32), persistent=False)
            self.register_buffer("V_k", V_k.to(device=device, dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("U_k", None, persistent=False)
            self.register_buffer("V_k", None, persistent=False)

    def get_effective_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (A_eff, B_eff) in FP32. For OPLoRA returns projected factors (A_tilde, B_tilde)."""
        if not self.use_oplora:
            return self.lora_A, self.lora_B
        B_tilde = self.lora_B - torch.matmul(self.U_k, torch.matmul(self.U_k.transpose(0, 1), self.lora_B))
        A_tilde = self.lora_A - torch.matmul(torch.matmul(self.lora_A, self.V_k), self.V_k.transpose(0, 1))
        return A_tilde, B_tilde

    def get_delta_weight(self) -> torch.Tensor:
        """Compute full correction matrix Delta W = (alpha / r) B_eff @ A_eff in FP32."""
        A_eff, B_eff = self.get_effective_factors()
        return self.scaling * torch.matmul(B_eff, A_eff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_dropped = self.lora_dropout(x).to(torch.float32)
        A_eff, B_eff = self.get_effective_factors()
        lora_out = torch.matmul(torch.matmul(x_dropped, A_eff.transpose(0, 1)), B_eff.transpose(0, 1))
        return base_out + (self.scaling * lora_out).to(base_out.dtype)

def inject_adapters(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    use_oplora: bool = False,
    protection_rank_k: int = 128,
    target_modules: Tuple[str, ...] = TASK1_TARGET_MODULES,
    init_seed: int = 6304,
    precomputed_svd_dict: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Dict[str, float]:
    """Freeze backbone and wrap target linear layers with LoRALinear / OPLoRA."""
    for param in model.parameters():
        param.requires_grad_(False)

    t0 = time.perf_counter()
    basis_bytes = 0
    replaced_count = 0

    named_modules = dict(model.named_modules())
    for full_name, module in list(named_modules.items()):
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.split(".")[-1]
        if leaf_name not in target_modules:
            continue

        parent_name = ".".join(full_name.split(".")[:-1])
        parent = named_modules[parent_name] if parent_name else model
        bases = precomputed_svd_dict.get(full_name) if precomputed_svd_dict else None

        wrapped = LoRALinear(
            base_layer=module,
            r=r,
            alpha=alpha,
            dropout=dropout,
            use_oplora=use_oplora,
            protection_rank_k=protection_rank_k,
            precomputed_bases=bases,
            init_seed=init_seed,
            layer_tag=full_name,
        )
        if use_oplora and wrapped.U_k is not None:
            basis_bytes += (wrapped.U_k.numel() + wrapped.V_k.numel()) * 4

        setattr(parent, leaf_name, wrapped)
        replaced_count += 1

    svd_time_sec = time.perf_counter() - t0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "replaced_layers": replaced_count,
        "trainable_params": trainable_params,
        "svd_preprocess_time_sec": svd_time_sec if use_oplora else 0.0,
        "basis_storage_mb": basis_bytes / (1024 ** 2),
    }

@torch.no_grad()
def export_adapter_state(model: nn.Module, save_path: str | Path, export_projected: bool = True) -> Dict[str, float]:
    """Export adapter weights (using projected factors A_tilde, B_tilde for OPLoRA)."""
    t0 = time.perf_counter()
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if export_projected and module.use_oplora:
                A_eff, B_eff = module.get_effective_factors()
                state[f"{name}.lora_A"] = A_eff.detach().cpu()
                state[f"{name}.lora_B"] = B_eff.detach().cpu()
            else:
                state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
                state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            state[f"{name}.scaling"] = torch.tensor(module.scaling, dtype=torch.float32)
    export_time_sec = time.perf_counter() - t0
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    size_mb = path.stat().st_size / (1024 ** 2)
    return {"export_time_sec": export_time_sec, "checkpoint_size_mb": size_mb}

@torch.no_grad()
def merge_adapters_into_backbone(model: nn.Module) -> nn.Module:
    """Merge LoRA/OPLoRA effective factors into base_layer weights in-place."""
    named_modules = dict(model.named_modules())
    for full_name, module in list(named_modules.items()):
        if isinstance(module, LoRALinear):
            delta_w = module.get_delta_weight().to(
                dtype=module.base_layer.weight.dtype, device=module.base_layer.weight.device
            )
            module.base_layer.weight.data.add_(delta_w)
            parent_name = ".".join(full_name.split(".")[:-1])
            leaf_name = full_name.split(".")[-1]
            parent = named_modules[parent_name] if parent_name else model
            setattr(parent, leaf_name, module.base_layer)
    return model

@torch.no_grad()
def load_exported_adapters_and_merge(
    base_model: nn.Module,
    adapter_ckpt_path: str | Path,
) -> nn.Module:
    """Load exported (projected) adapter factors into a fresh FP16 backbone copy and merge in-place."""
    state = torch.load(adapter_ckpt_path, map_location="cpu")
    named_modules = dict(base_model.named_modules())
    for key, val in state.items():
        if not key.endswith(".lora_A"):
            continue
        prefix = key[: -len(".lora_A")]
        A_eff = val.to(torch.float32)
        B_eff = state[f"{prefix}.lora_B"].to(torch.float32)
        scaling = float(state[f"{prefix}.scaling"].item())
        mod = named_modules[prefix]
        target_lin = mod.base_layer if isinstance(mod, LoRALinear) else mod
        delta_w = (scaling * torch.matmul(B_eff.to(target_lin.weight.device), A_eff.to(target_lin.weight.device))).to(
            target_lin.weight.dtype
        )
        target_lin.weight.data.add_(delta_w)
    return base_model

@torch.no_grad()
def verify_merged_vs_unmerged_logits(
    unmerged_model: nn.Module,
    merged_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Check merged versus unmerged logits on a small batch with dropout disabled."""
    unmerged_model.eval()
    merged_model.eval()
    out_u = unmerged_model(input_ids=input_ids, attention_mask=attention_mask)
    out_m = merged_model(input_ids=input_ids, attention_mask=attention_mask)
    logits_u = out_u.logits if hasattr(out_u, "logits") else out_u[0]
    logits_m = out_m.logits if hasattr(out_m, "logits") else out_m[0]
    diff = (logits_u.float() - logits_m.float()).abs()
    return {
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
    }
