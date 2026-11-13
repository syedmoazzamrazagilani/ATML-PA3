from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .config import Task3Config
except ImportError:
    from config import Task3Config


METHOD_LABELS = {
    "first_ans": "First-Answer (N=1)",
    "self_consistency": "Self-Consistency (MajVote)",
    "best_logp": "Best-of-N (Mean Log-Prob L)",
    "best_self_certainty": "Best-of-N (Self-Certainty K)",
    "rwc_self_certainty": "RWC (Self-Certainty K)",
    "shortest_length": "Shortest Rollout (-T)",
    "best_vision": "Best-of-N (Vision V)",
    "rwc_vision": "RWC (Vision V)",
    "best_visconf": "Best-of-N (VisConf Z)",
    "rwc_visconf": "RWC (VisConf Z)",
    "best_visconf_token": "Best-of-N (VisConf-Token)",
    "rwc_visconf_token": "RWC (VisConf-Token)",
}


def plot_step1_scaling_and_decomposition(
    results_by_dataset: Dict[str, Dict[str, Any]],
    cfg: Task3Config,
) -> List[Path]:
    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for ds_key, ds_res in results_by_dataset.items():
        df: pd.DataFrame = ds_res["step2_df"]
        sc_df = df[df["method"] == "self_consistency"].sort_values("N")
        fa_df = df[df["method"] == "first_ans"].sort_values("N")

        budgets = sc_df["N"].to_numpy()
        x_pos = np.arange(len(budgets))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

        ax0 = axes[0]
        ax0.plot(
            x_pos,
            sc_df["oracle_coverage"] * 100.0,
            marker="o",
            linewidth=2.2,
            linestyle="--",
            color="#2b5c8f",
            label="Oracle Pass@N Coverage",
        )
        ax0.plot(
            x_pos,
            sc_df["accuracy"] * 100.0,
            marker="s",
            linewidth=2.2,
            color="#d95f02",
            label="Self-Consistency (MajVote)",
        )
        ax0.plot(
            x_pos,
            fa_df["accuracy"] * 100.0,
            marker="^",
            linewidth=1.8,
            linestyle=":",
            color="#7570b3",
            label="First-Answer Baseline",
        )
        ax0.set_xticks(x_pos)
        ax0.set_xticklabels([str(n) for n in budgets])
        ax0.set_xlabel("Rollout Budget N")
        ax0.set_ylabel("Accuracy / Coverage (%)")
        ax0.set_title(f"{ds_key.upper()} — Step 1 Scaling & Oracle Gap")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="best", frameon=True)

        ax1 = axes[1]
        samp_fail = sc_df["sampling_failure_rate"].to_numpy() * 100.0
        sel_fail = sc_df["selection_failure_rate"].to_numpy() * 100.0

        ax1.bar(
            x_pos,
            samp_fail,
            width=0.55,
            color="#8da0cb",
            label="Sampling Failure (No Correct Candidate)",
        )
        ax1.bar(
            x_pos,
            sel_fail,
            width=0.55,
            bottom=samp_fail,
            color="#fc8d62",
            label="Selection Failure (Correct Missed)",
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([str(n) for n in budgets])
        ax1.set_xlabel("Rollout Budget N")
        ax1.set_ylabel("Total Error Rate (%)")
        ax1.set_title(f"{ds_key.upper()} — Failure Decomposition & Answer Groups")
        ax1.grid(True, axis="y", alpha=0.3)

        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            x_pos,
            sc_df["distinct_answer_groups"].to_numpy(),
            color="#1b9e77",
            marker="D",
            linewidth=2.0,
            label="Distinct Answer Groups",
        )
        ax1_twin.set_ylabel("Mean Distinct Answer Groups")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            out_p = fig_dir / f"step1_{ds_key}_decomposition.{ext}"
            fig.savefig(out_p, dpi=220)
            saved_paths.append(out_p)
        plt.close(fig)

    return saved_paths


def plot_step2_aggregation_comparison(
    results_by_dataset: Dict[str, Dict[str, Any]],
    cfg: Task3Config,
) -> List[Path]:
    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for ds_key, ds_res in results_by_dataset.items():
        df: pd.DataFrame = ds_res["step2_df"]
        boot: Dict[str, Any] = ds_res["step2_bootstrap"]

        budgets = sorted(df["N"].unique())
        x_pos = np.arange(len(budgets))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

        ax0 = axes[0]
        plot_methods = [
            "self_consistency",
            "best_logp",
            "best_self_certainty",
            "rwc_self_certainty",
            "shortest_length",
        ]
        if ds_key.lower() == "mathvista":
            plot_methods.extend(["best_vision", "best_visconf", "rwc_visconf"])

        for m in plot_methods:
            sub = df[df["method"] == m].sort_values("N")
            if sub.empty:
                continue
            linestyle = "-" if "rwc" in m or m == "self_consistency" else "--"
            marker = "o" if "rwc" in m else "s"
            ax0.plot(
                x_pos,
                sub["accuracy"].to_numpy() * 100.0,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                label=METHOD_LABELS.get(m, m),
            )

        ax0.set_xticks(x_pos)
        ax0.set_xticklabels([str(n) for n in budgets])
        ax0.set_xlabel("Rollout Budget N")
        ax0.set_ylabel("Accuracy (%)")
        ax0.set_title(f"{ds_key.upper()} — Step 2 Aggregation Accuracy vs N")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="lower right", fontsize=8, frameon=True)

        ax1 = axes[1]
        n32_keys = [k for k in boot.keys() if k.startswith("N32_")]
        labels, means, ci_lows, ci_highs = [], [], [], []

        for k in n32_keys:
            item = boot[k]
            clean_label = k.replace("N32_", "").replace("_vs_", " vs ")
            labels.append(clean_label)

            m_raw = next(
                (
                    item[key]
                    for key in ("mean_diff", "delta_mean", "diff_mean", "mean", "estimate", "diff")
                    if key in item
                ),
                0.0,
            )
            lo_raw = next(
                (
                    item[key]
                    for key in ("ci_low", "ci_95_low", "ci_lower", "lower", "low")
                    if key in item
                ),
                m_raw,
            )
            hi_raw = next(
                (
                    item[key]
                    for key in ("ci_high", "ci_95_high", "ci_upper", "upper", "high")
                    if key in item
                ),
                m_raw,
            )

            m_val = float(m_raw) * 100.0
            lo_val = float(lo_raw) * 100.0
            hi_val = float(hi_raw) * 100.0
            lo_val, hi_val = min(lo_val, m_val, hi_val), max(lo_val, m_val, hi_val)

            means.append(m_val)
            ci_lows.append(max(0.0, m_val - lo_val))
            ci_highs.append(max(0.0, hi_val - m_val))

        y_pos = np.arange(len(labels))
        ax1.errorbar(
            means,
            y_pos,
            xerr=[ci_lows, ci_highs],
            fmt="o",
            color="#1f78b4",
            ecolor="#33a02c",
            elinewidth=2.0,
            capsize=4,
        )
        ax1.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel("Paired Accuracy Difference at N=32 (% points, 95% CI)")
        ax1.set_title(f"{ds_key.upper()} — Paired Bootstrap 95% CIs (N=32)")
        ax1.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            out_p = fig_dir / f"step2_{ds_key}_aggregation.{ext}"
            fig.savefig(out_p, dpi=220)
            saved_paths.append(out_p)
        plt.close(fig)

    return saved_paths


def plot_step3_quartile_and_signals(
    results_by_dataset: Dict[str, Dict[str, Any]],
    cfg: Task3Config,
) -> List[Path]:
    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for ds_key, ds_res in results_by_dataset.items():
        s3 = ds_res["step3"]["quartile_analysis"]
        score_names = list(s3.keys())
        top_accs = [s3[s]["top_quartile_acc"] * 100.0 for s in score_names]
        bot_accs = [s3[s]["bottom_quartile_acc"] * 100.0 for s in score_names]
        diffs = [s3[s]["mean_quartile_diff"] * 100.0 for s in score_names]
        errs = [max(0.0, float(s3[s]["std_err_diff"])) * 1.96 * 100.0 for s in score_names]

        x = np.arange(len(score_names))
        width = 0.35

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

        ax0 = axes[0]
        ax0.bar(x - width / 2, top_accs, width, label="Top Quartile (Top 8 of 32)", color="#2ca25f")
        ax0.bar(x + width / 2, bot_accs, width, label="Bottom Quartile (Bottom 8 of 32)", color="#de2d26")
        ax0.set_xticks(x)
        ax0.set_xticklabels(score_names)
        ax0.set_ylabel("Rollout Accuracy (%)")
        ax0.set_title(f"{ds_key.upper()} — Step 3 Top vs Bottom Quartile Accuracy")
        ax0.grid(True, axis="y", alpha=0.3)
        ax0.legend(loc="best", frameon=True)

        ax1 = axes[1]
        ax1.bar(x, diffs, width=0.5, yerr=errs, capsize=4, color="#3182bd")
        ax1.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax1.set_xticks(x)
        ax1.set_xticklabels(score_names)
        ax1.set_ylabel("Top - Bottom Quartile Spread (% pts)")
        ax1.set_title(f"{ds_key.upper()} — Within-Question Signal Discriminability")
        ax1.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            out_p = fig_dir / f"step3_{ds_key}_quartiles.{ext}"
            fig.savefig(out_p, dpi=220)
            saved_paths.append(out_p)
        plt.close(fig)

    return saved_paths


def plot_step4_live_comparison(
    step4_result: Dict[str, Any],
    cfg: Task3Config,
) -> List[Path]:
    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    if "strategy_summary" not in step4_result:
        return saved_paths

    df = pd.DataFrame(step4_result["strategy_summary"])
    trans = step4_result.get("refinement_transitions", {})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax0 = axes[0]
    x = np.arange(len(df))
    w = 0.28
    ax0.bar(x - w, df["accuracy"] * 100.0, width=w, label="Accuracy (%)", color="#2b8cbe")
    ax0.bar(x, df["extraction_rate"] * 100.0, width=w, label="Boxed Extraction Rate (%)", color="#7bccc4")
    ax0.bar(x + w, df["truncation_rate"] * 100.0, width=w, label="Truncation Rate (%)", color="#e34a33")
    ax0.set_xticks(x)
    ax0.set_xticklabels(df["strategy"].tolist(), rotation=15, ha="right", fontsize=8)
    ax0.set_ylabel("Rate (%)")
    ax0.set_title("MATH Step 4 Live — Accuracy, Extraction & Truncation")
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend(loc="best", fontsize=8, frameon=True)

    ax1 = axes[1]
    t_keys = ["wrong_to_correct_count", "correct_to_wrong_count", "correct_to_correct_count", "wrong_to_wrong_count"]
    t_labels = ["Wrong -> Correct", "Correct -> Wrong", "Correct -> Correct", "Wrong -> Wrong"]
    t_vals = [int(trans.get(k, 0)) for k in t_keys]
    colors = ["#31a354", "#de2d26", "#6baed6", "#969696"]
    ax1.bar(np.arange(len(t_labels)), t_vals, width=0.5, color=colors)
    ax1.set_xticks(np.arange(len(t_labels)))
    ax1.set_xticklabels(t_labels, rotation=12, fontsize=9)
    ax1.set_ylabel("Number of Questions (out of 50)")
    ax1.set_title("MATH Step 4 Live — Sequential Refinement Transitions")
    ax1.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out_p = fig_dir / f"step4_math_live_comparison.{ext}"
        fig.savefig(out_p, dpi=220)
        saved_paths.append(out_p)
    plt.close(fig)

    return saved_paths


def plot_step5_adaptive_pareto(
    results_by_dataset: Dict[str, Dict[str, Any]],
    cfg: Task3Config,
) -> List[Path]:
    fig_dir = cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for ds_key, ds_res in results_by_dataset.items():
        s2_df: pd.DataFrame = ds_res["step2_df"]
        sc_df = s2_df[s2_df["method"] == "self_consistency"].sort_values("N")
        s5 = ds_res["step5"]["eval_results"]

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

        ax0 = axes[0]
        ax0.plot(
            sc_df["mean_token_cost"],
            sc_df["accuracy"] * 100.0,
            marker="o",
            linewidth=2.0,
            color="#636363",
            label="Fixed-N Self-Consistency",
        )
        for _, row in sc_df.iterrows():
            ax0.annotate(
                f"N={int(row['N'])}",
                (row["mean_token_cost"], row["accuracy"] * 100.0),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

        agree_res = s5["agreement"]
        qa_res = s5["quality_aware"]
        ax0.scatter(
            [agree_res["mean_token_cost"]],
            [agree_res["accuracy"] * 100.0],
            s=110,
            marker="^",
            color="#e6550d",
            zorder=5,
            label=f"Agreement Adaptive (tau={agree_res['tau']:.3g})",
        )
        ax0.scatter(
            [qa_res["mean_token_cost"]],
            [qa_res["accuracy"] * 100.0],
            s=130,
            marker="*",
            color="#31a354",
            zorder=6,
            label=f"Quality-Aware Adaptive (tau={qa_res['tau']:.3g}, gamma={qa_res['gamma']:.3g})",
        )
        ax0.set_xlabel("Mean Token Cost per Question (Held-out Split)")
        ax0.set_ylabel("Accuracy (%)")
        ax0.set_title(f"{ds_key.upper()} — Step 5 Accuracy vs Compute Frontier")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="lower right", fontsize=8, frameon=True)

        ax1 = axes[1]
        checkpoints = [str(c) for c in list(cfg.adaptive_checkpoints) + [32]]
        agree_probs = [agree_res["stopping_distribution"].get(k, 0.0) * 100.0 for k in checkpoints]
        qa_probs = [qa_res["stopping_distribution"].get(k, 0.0) * 100.0 for k in checkpoints]

        x = np.arange(len(checkpoints))
        w = 0.35
        ax1.bar(x - w / 2, agree_probs, width=w, label="Agreement Controller", color="#fdae6b")
        ax1.bar(x + w / 2, qa_probs, width=w, label="Quality-Aware Controller", color="#74c476")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"n={k}" for k in checkpoints])
        ax1.set_ylabel("Stopping Frequency (%)")
        ax1.set_title(f"{ds_key.upper()} — Adaptive Stopping Checkpoint Distribution")
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.legend(loc="best", frameon=True)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            out_p = fig_dir / f"step5_{ds_key}_adaptive.{ext}"
            fig.savefig(out_p, dpi=220)
            saved_paths.append(out_p)
        plt.close(fig)

    return saved_paths


def export_all_tables(
    results_by_dataset: Dict[str, Dict[str, Any]],
    step4_result: Optional[Dict[str, Any]],
    cfg: Task3Config,
) -> Path:
    table_dir = cfg.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = ["# Task 3 Test-Time Compute Evaluation Summary\n"]

    for ds_key, ds_res in results_by_dataset.items():
        md_lines.append(f"## Dataset: `{ds_key.upper()}` (Held-Out Evaluation Split, 800 Questions)\n")

        df_s2: pd.DataFrame = ds_res["step2_df"]
        df_n32 = df_s2[df_s2["N"] == 32].copy()
        df_n32.to_csv(table_dir / f"{ds_key}_n32_methods.csv", index=False)
        try:
            df_n32.to_latex(table_dir / f"{ds_key}_n32_methods.tex", index=False, float_format="%.4f")
        except Exception:
            pass

        md_lines.append("### Step 1 & 2: Aggregation Performance at N = 32\n")
        md_lines.append(
            "| Method | Accuracy (%) | Oracle Coverage (%) | Sampling Fail (%) | Selection Fail (%) | Distinct Groups | Mean Tokens |"
        )
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in df_n32.iterrows():
            md_lines.append(
                f"| `{r['method']}` | {r['accuracy']*100:.2f} | {r['oracle_coverage']*100:.2f} | "
                f"{r['sampling_failure_rate']*100:.2f} | {r['selection_failure_rate']*100:.2f} | "
                f"{r['distinct_answer_groups']:.2f} | {r['mean_token_cost']:.1f} |"
            )
        md_lines.append("")

        s3_q = ds_res["step3"]["quartile_analysis"]
        md_lines.append("### Step 3: Within-Question Top vs Bottom Quartile Analysis (N = 32)\n")
        md_lines.append(
            "| Signal | Top-8 Acc (%) | Bottom-8 Acc (%) | Spread (% pts) | Within-Q Corr w/ Length |"
        )
        md_lines.append("|---|---:|---:|---:|---:|")
        for sig, vals in s3_q.items():
            corr_str = (
                f"{vals['mean_within_q_corr_with_length']:.3f}"
                if vals["mean_within_q_corr_with_length"] is not None
                else "N/A"
            )
            md_lines.append(
                f"| `{sig}` | {vals['top_quartile_acc']*100:.2f} | "
                f"{vals['bottom_quartile_acc']*100:.2f} | "
                f"{vals['mean_quartile_diff']*100:+.2f} | {corr_str} |"
            )
        md_lines.append("")

        s5_eval = ds_res["step5"]["eval_results"]
        md_lines.append("### Step 5: Held-Out Adaptive Sampling vs Fixed-32 Self-Consistency\n")
        md_lines.append(
            "| Controller | Selected (tau, gamma) | Accuracy (%) | Mean Rollouts | Mean Tokens | Token Savings vs Fixed-32 (%) |"
        )
        md_lines.append("|---|---:|---:|---:|---:|---:|")
        for c_key in ("fixed_32", "agreement", "quality_aware"):
            item = s5_eval[c_key]
            md_lines.append(
                f"| `{c_key}` | ({item['tau']:.3g}, {item['gamma']:.3g}) | "
                f"{item['accuracy']*100:.2f} | {item['mean_rollouts']:.2f} | "
                f"{item['mean_token_cost']:.1f} | {item['token_savings_vs_fixed32']*100:.2f} |"
            )
        md_lines.append("")

    if step4_result is not None and "strategy_summary" in step4_result:
        md_lines.append("## Step 4: Live Generation on 50 MATH Evaluation Questions (`Qwen/Qwen2.5-3B-Instruct`)\n")
        md_lines.append(
            "| Strategy | Accuracy (%) | Mean In Tokens | Mean Out Tokens | Runtime (s/q) | Peak GPU (MB) | Extraction (%) | Truncation (%) |"
        )
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in step4_result["strategy_summary"]:
            md_lines.append(
                f"| `{row['strategy']}` | {row['accuracy']*100:.2f} | "
                f"{row['mean_input_tokens']:.1f} | {row['mean_output_tokens']:.1f} | "
                f"{row['mean_runtime_sec']:.2f} | {row['peak_gpu_mb']:.0f} | "
                f"{row['extraction_rate']*100:.1f} | {row['truncation_rate']*100:.1f} |"
            )
        md_lines.append("")
        tr = step4_result.get("refinement_transitions", {})
        md_lines.append(
            f"**Sequential Refinement Transitions:** "
            f"Wrong->Correct = {tr.get('wrong_to_correct_count', 0)}, "
            f"Correct->Wrong = {tr.get('correct_to_wrong_count', 0)}, "
            f"Correct->Correct = {tr.get('correct_to_correct_count', 0)}, "
            f"Wrong->Wrong = {tr.get('wrong_to_wrong_count', 0)} "
            f"(Net Accuracy Change: {tr.get('net_accuracy_gain', 0.0)*100:+.2f} pp).\n"
        )

    report_path = table_dir / "task3_executive_summary.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return report_path


def generate_all_plots_and_tables(
    results_by_dataset: Dict[str, Dict[str, Any]],
    step4_result: Optional[Dict[str, Any]],
    cfg: Task3Config,
) -> Dict[str, Any]:
    figs: List[Path] = []
    figs.extend(plot_step1_scaling_and_decomposition(results_by_dataset, cfg))
    figs.extend(plot_step2_aggregation_comparison(results_by_dataset, cfg))
    figs.extend(plot_step3_quartile_and_signals(results_by_dataset, cfg))
    if step4_result is not None:
        figs.extend(plot_step4_live_comparison(step4_result, cfg))
    figs.extend(plot_step5_adaptive_pareto(results_by_dataset, cfg))

    report_path = export_all_tables(results_by_dataset, step4_result, cfg)
    return {
        "figures": [str(p) for p in figs],
        "executive_summary_md": str(report_path),
    }
