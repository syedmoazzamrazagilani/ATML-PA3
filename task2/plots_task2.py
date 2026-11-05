from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt

from common.plot_style import apply_neurips_plot_style


def plot_task2_learning_curves(
    train_history: List[Dict[str, Any]],
    dev_history: List[Dict[str, Any]],
    save_path: str | Path,
    stage_boundaries: Tuple[int, int] = (2250, 3750),
) -> None:
    """
    Plot 4-panel learning and development grounding curves across the 3 curriculum stages:
      (a) Training Loss with stage boundaries
      (b) Development Caption Loss
      (c) Development Reference-Answer NLL (Correct vs Complementary vs No-Image)
      (d) Image-Swap Penalty Delta_NLL = NLL(partner) - NLL(correct)
    """
    apply_neurips_plot_style()
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.8))

    # Panel (a): Training loss
    ax0 = axes[0]
    if train_history:
        steps = [r["update"] for r in train_history]
        losses = [r["loss"] for r in train_history]
        ax0.plot(steps, losses, color="#1F78B4", alpha=0.25, linewidth=0.7)
        win = min(25, max(1, len(losses) // 15))
        if win > 1:
            sm = np.convolve(losses, np.ones(win) / win, mode="valid")
            ax0.plot(steps[win - 1 :], sm, color="#1F78B4", linewidth=1.5, label="Train Loss")
        else:
            ax0.plot(steps, losses, color="#1F78B4", linewidth=1.5, label="Train Loss")
    for b in stage_boundaries:
        ax0.axvline(b, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
    ax0.set_title("(a) Curriculum Training Loss")
    ax0.set_xlabel("Update")
    ax0.set_ylabel("Example-Weighted CE Loss")

    # Panel (b): Dev Caption Reference Loss
    ax1 = axes[1]
    if dev_history:
        d_steps = [r["update"] for r in dev_history]
        cap_loss = [r["dev_caption_loss"] for r in dev_history]
        ax1.plot(d_steps, cap_loss, marker="o", markersize=3.5, color="#33A02C", linewidth=1.5)
    for b in stage_boundaries:
        ax1.axvline(b, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_title("(b) Dev Caption Reference Loss")
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Caption CE Loss")

    # Panel (c): Dev Answer NLL
    ax2 = axes[2]
    if dev_history:
        d_steps = [r["update"] for r in dev_history]
        g_corr = [r["grounding_dev"]["nll_correct"] for r in dev_history]
        g_comp = [r["grounding_dev"]["nll_complementary"] for r in dev_history]
        v_corr = [r["natural_vqa_dev"]["nll_correct"] for r in dev_history]
        v_comp = [r["natural_vqa_dev"]["nll_complementary"] for r in dev_history]
        ax2.plot(d_steps, g_corr, "-o", markersize=3, color="#1F78B4", label="Grounding (Correct)")
        ax2.plot(d_steps, g_comp, "--s", markersize=3, color="#1F78B4", alpha=0.65, label="Grounding (Swap)")
        ax2.plot(d_steps, v_corr, "-o", markersize=3, color="#E31A1C", label="VQAv2 (Correct)")
        ax2.plot(d_steps, v_comp, "--s", markersize=3, color="#E31A1C", alpha=0.65, label="VQAv2 (Swap)")
    for b in stage_boundaries:
        ax2.axvline(b, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
    ax2.set_title("(c) Dev Reference-Answer NLL")
    ax2.set_xlabel("Update")
    ax2.set_ylabel("Token-Avg Answer NLL")
    ax2.legend(loc="best", fontsize=6.5)

    # Panel (d): Image-Swap Penalty Delta_NLL
    ax3 = axes[3]
    if dev_history:
        d_steps = [r["update"] for r in dev_history]
        g_dnll = [r["grounding_dev"]["delta_nll"] for r in dev_history]
        v_dnll = [r["natural_vqa_dev"]["delta_nll"] for r in dev_history]
        ax3.axhline(0.0, color="#666666", linestyle=":", linewidth=0.8)
        ax3.plot(d_steps, g_dnll, "-o", markersize=3.5, color="#1F78B4", label="Grounding Dev")
        ax3.plot(d_steps, v_dnll, "-s", markersize=3.5, color="#E31A1C", label="Natural VQAv2 Dev")
    for b in stage_boundaries:
        ax3.axvline(b, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
    ax3.set_title(r"(d) Swap Penalty $\Delta_{\mathrm{NLL}}$")
    ax3.set_xlabel("Update")
    ax3.set_ylabel(r"$\mathrm{NLL}(I_{\mathrm{partner}}) - \mathrm{NLL}(I_{\mathrm{correct}})$")
    ax3.legend(loc="best", fontsize=7)

    plt.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)


def plot_task2_final_grounding_comparison(
    final_results: Dict[str, Any],
    save_path: str | Path,
) -> None:
    """
    Plot 2-panel grouped bar chart comparing Correct, Complementary Swap, Mismatched,
    and No-Image conditions on Grounding Test (left) and Natural VQAv2 Test (right),
    including 95% bootstrap confidence intervals.
    """
    apply_neurips_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    conds = ["correct", "complementary", "mismatched", "no_image"]
    cond_labels = ["Correct", "Comp. Swap", "Mismatched", "No Image"]
    x = np.arange(len(conds))
    width = 0.25

    for idx, (suite_key, title) in enumerate([
        ("grounding_test", "(a) Annotation-Derived Grounding Test"),
        ("natural_vqa_test", "(b) Natural VQAv2 Test"),
    ]):
        ax = axes[idx]
        if suite_key not in final_results:
            continue
        s_data = final_results[suite_key]["conditions"]
        ci_data = final_results[suite_key]["bootstrap_95ci"]["per_condition"]

        acc = [s_data[c]["overall_accuracy"] for c in conds]
        pacc = [s_data[c]["pair_accuracy"] for c in conds]
        ident = [s_data[c]["identical_answer_rate"] for c in conds]

        acc_err = [
            [
                acc[i] - ci_data[c]["acc_ci95"][0],
                ci_data[c]["acc_ci95"][1] - acc[i],
            ]
            for i, c in enumerate(conds)
        ]
        acc_err_T = np.array(acc_err).T

        ax.bar(x - width, acc, width, yerr=acc_err_T, capsize=2.5, label="Answer Accuracy (%)", color="#1F78B4")
        ax.bar(x, pacc, width, label="Pair Accuracy (%)", color="#33A02C")
        ax.bar(x + width, ident, width, label="Identical-Answer Rate (%)", color="#FB9A99")

        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", fontsize=7.5)

    plt.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
