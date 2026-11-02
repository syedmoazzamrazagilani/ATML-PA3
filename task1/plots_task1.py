from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

from common.plot_style import apply_neurips_plot_style, PALETTE


def plot_task1_learning_curves(
    histories: Dict[str, List[Dict[str, Any]]],
    save_path: str | Path,
) -> None:
    """
    Generate combined SFT-loss plot for the five runs (left) and L2-SP penalty / objective
    decomposition (right), saved as 300-DPI PDF and PNG.
    """
    apply_neurips_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1))

    ax0 = axes[0]
    label_map = {
        "full_ft": ("Full FT", PALETTE["full_ft"]),
        "l2_sp": ("Full FT + L2-SP", PALETTE["l2_sp"]),
        "lora_4": ("LoRA (r=4)", PALETTE["lora_4"]),
        "lora_16": ("LoRA (r=16)", PALETTE["lora_16"]),
        "oplora_16": ("OPLoRA (r=16, k=128)", PALETTE["oplora_16"]),
    }
    for key, (label, color) in label_map.items():
        if key not in histories or not histories[key]:
            continue
        steps = [r["update"] for r in histories[key]]
        sft_losses = [r["sft_loss"] for r in histories[key]]
        ax0.plot(steps, sft_losses, color=color, alpha=0.22, linewidth=0.8)
        window = min(15, max(1, len(sft_losses) // 10))
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(sft_losses, kernel, mode="valid")
            ax0.plot(
                steps[window - 1 :], smoothed, color=color, label=label, linewidth=1.6
            )
        else:
            ax0.plot(steps, sft_losses, color=color, label=label, linewidth=1.6)

    ax0.set_title("(a) SFT Loss Across Optimizer Updates")
    ax0.set_xlabel("Optimizer Update")
    ax0.set_ylabel("Token-Normalized SFT Loss")
    ax0.legend(loc="upper right", frameon=True)

    ax1 = axes[1]
    if "l2_sp" in histories and histories["l2_sp"]:
        steps_l2 = [r["update"] for r in histories["l2_sp"]]
        sft_l2 = [r["sft_loss"] for r in histories["l2_sp"]]
        reg_l2 = [r["lambda_omega"] for r in histories["l2_sp"]]
        tot_l2 = [r["total_loss"] for r in histories["l2_sp"]]
        ax1.plot(
            steps_l2,
            sft_l2,
            color=PALETTE["l2_sp"],
            linestyle="--",
            label="L2-SP: SFT Loss",
            linewidth=1.3,
        )
        ax1.plot(
            steps_l2,
            reg_l2,
            color="#FF7F00",
            linestyle=":",
            label=r"L2-SP: $\lambda \Omega(\theta, \theta_0)$",
            linewidth=1.5,
        )
        ax1.plot(
            steps_l2,
            tot_l2,
            color=PALETTE["l2_sp"],
            linestyle="-",
            label="L2-SP: Total Objective",
            linewidth=1.6,
        )
    if "full_ft" in histories and histories["full_ft"]:
        steps_ft = [r["update"] for r in histories["full_ft"]]
        sft_ft = [r["sft_loss"] for r in histories["full_ft"]]
        ax1.plot(
            steps_ft,
            sft_ft,
            color=PALETTE["full_ft"],
            alpha=0.75,
            label="Full FT: SFT Loss",
            linewidth=1.3,
        )

    ax1.set_title(r"(b) Full FT vs. L2-SP ($\lambda=10^{-3}$) Objective")
    ax1.set_xlabel("Optimizer Update")
    ax1.set_ylabel("Loss / Penalty Value")
    ax1.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_task1_diagnostics(
    weight_diag: Dict[str, Dict[int, Dict[str, float]]],
    act_diag: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]],
    save_path: str | Path,
) -> None:
    """
    Create 3-panel diagnostic figure across layers 0, 13, 27:
      (a) Subspace interference ratio rho_k
      (b) Relative update magnitude eta
      (c) Activation perturbation ratio R(z) on Math (GSM8K) vs Non-Math (HellaSwag)
    """
    apply_neurips_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    layers = [0, 13, 27]
    x = np.arange(len(layers))
    variants = ["full_ft_mid", "full_ft", "l2_sp", "lora_4", "lora_16", "oplora_16"]
    labels = ["Full FT (Mid)", "Full FT", "L2-SP", "LoRA-4", "LoRA-16", "OPLoRA-16"]
    width = 0.13

    ax0 = axes[0]
    for idx, (v, lbl) in enumerate(zip(variants, labels)):
        if v not in weight_diag:
            continue
        vals = [
            weight_diag[v].get(l, weight_diag[v].get(str(l), {})).get("rho_k", 0.0)
            for l in layers
        ]
        ax0.bar(
            x + (idx - 2.5) * width,
            vals,
            width,
            label=lbl,
            color=PALETTE.get(v, "#555555"),
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"Layer {l}" for l in layers])
    ax0.set_title(r"(a) Subspace Interference $\rho_{128}$")
    ax0.set_ylabel(r"$\rho_k = \|U_k U_k^\top \Delta W\|_F^2 / \|\Delta W\|_F^2$")
    ax0.legend(loc="best", fontsize=7)

    ax1 = axes[1]
    for idx, (v, lbl) in enumerate(zip(variants, labels)):
        if v not in weight_diag:
            continue
        vals = [
            weight_diag[v].get(l, weight_diag[v].get(str(l), {})).get("eta", 0.0)
            for l in layers
        ]
        ax1.bar(
            x + (idx - 2.5) * width,
            vals,
            width,
            label=lbl,
            color=PALETTE.get(v, "#555555"),
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Layer {l}" for l in layers])
    ax1.set_title(r"(b) Relative Update Magnitude $\eta$")
    ax1.set_ylabel(r"$\eta = \|\Delta W\|_F / \|W_0\|_F$")

    ax2 = axes[2]
    math_means, nonmath_means, v_labels = [], [], []
    for v, lbl in zip(variants, labels):
        if v not in act_diag:
            continue
        v_labels.append(lbl)
        m_vals = [
            act_diag[v]
            .get(l, act_diag[v].get(str(l), {}))
            .get("math", {})
            .get("mean_R", 0.0)
            for l in layers
        ]
        nm_vals = [
            act_diag[v]
            .get(l, act_diag[v].get(str(l), {}))
            .get("non_math", {})
            .get("mean_R", 0.0)
            for l in layers
        ]
        math_means.append(float(np.mean(m_vals)))
        nonmath_means.append(float(np.mean(nm_vals)))

    xv = np.arange(len(v_labels))
    ax2.bar(xv - 0.18, math_means, 0.35, label="GSM8K (Math)", color="#1F78B4")
    ax2.bar(xv + 0.18, nonmath_means, 0.35, label="HellaSwag (Non-Math)", color="#E31A1C")
    ax2.set_xticks(xv)
    ax2.set_xticklabels(v_labels, rotation=25, ha="right")
    ax2.set_title(r"(c) Activation Perturbation $R(z)$")
    ax2.set_ylabel(r"Mean $\|\Delta W z\|_2 / \|W_0 z\|_2$")
    ax2.legend(loc="best", fontsize=7.5)

    plt.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
