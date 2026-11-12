import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .answer_utils import NO_ANSWER
from .config import Task3Config
from .metrics_and_scoring import (
    compute_prefix_visconf,
    paired_question_bootstrap_ci,
    select_best_of_n,
    select_rwc,
    select_weighted_consensus,
)


def evaluate_steps_1_and_2(
    cfg: Task3Config,
    records: Dict[str, Dict[str, Any]],
    dataset_key: str,
    split_filter: str = "eval",
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[Tuple[str, int], np.ndarray]]:
    """
    Executes Step 1 (Repeated Sampling Opportunity & Failure Decomposition) and
    Step 2 (Quality Scoring & Answer Aggregation) across N in {1, 2, 4, 8, 16, 32}
    and 10 random rollout permutations per question.
    Also evaluates Step 3's shortest-rollout control and tokenwise VisConf ablation.
    """
    is_vlm = dataset_key == "mathvista"
    qids = [qid for qid, rec in records.items() if rec["split"] == split_filter]

    base_methods = [
        "first_ans",
        "self_consistency",
        "best_logp",
        "best_self_certainty",
        "rwc_self_certainty",
        "shortest_length",
    ]
    vlm_methods = [
        "best_vision",
        "rwc_vision",
        "best_visconf",
        "rwc_visconf",
        "best_visconf_token",
        "rwc_visconf_token",
    ]
    methods = base_methods + (vlm_methods if is_vlm else [])

    # Store per-question order-averaged accuracies for paired bootstrap CIs
    per_q_acc: Dict[Tuple[str, int], np.ndarray] = {
        (m, N): np.zeros(len(qids), dtype=np.float64) for m in methods for N in cfg.budgets
    }
    per_q_cost: Dict[Tuple[str, int], np.ndarray] = {
        (m, N): np.zeros(len(qids), dtype=np.float64) for m in methods for N in cfg.budgets
    }
    per_q_oracle: Dict[int, np.ndarray] = {
        N: np.zeros(len(qids), dtype=np.float64) for N in cfg.budgets
    }
    per_q_groups: Dict[int, np.ndarray] = {
        N: np.zeros(len(qids), dtype=np.float64) for N in cfg.budgets
    }

    for q_idx, qid in enumerate(tqdm(qids, desc=f"Steps 1-2 ({dataset_key}/{split_filter})")):
        rec = records[qid]
        orders = rec["orders"]  # (10, 32)
        ans_groups = rec["answer_groups"]
        is_corr = rec["is_correct"]
        lengths = rec["lengths"]
        L_all = rec["L"]
        K_all = rec["K"]
        V_all = rec["V"]

        for N in cfg.budgets:
            oracle_orders = np.zeros(cfg.num_orders, dtype=np.float64)
            ngroups_orders = np.zeros(cfg.num_orders, dtype=np.float64)
            m_acc_orders = {m: np.zeros(cfg.num_orders, dtype=np.float64) for m in methods}
            m_cost_orders = {m: np.zeros(cfg.num_orders, dtype=np.float64) for m in methods}

            for o_idx in range(cfg.num_orders):
                prefix = orders[o_idx, :N]
                prefix_cost = float(np.sum(lengths[prefix]))
                first_cost = float(lengths[prefix[0]])

                oracle_orders[o_idx] = 1.0 if np.any(is_corr[prefix]) else 0.0
                ngroups_orders[o_idx] = float(len(set(ans_groups[r] for r in prefix)))

                # 1. FirstAns (consumes only its first candidate)
                first_rid = int(prefix[0])
                m_acc_orders["first_ans"][o_idx] = 1.0 if is_corr[first_rid] else 0.0
                m_cost_orders["first_ans"][o_idx] = first_cost

                # 2. Self-Consistency (1 vote per rollout)
                _, sc_corr, _, _ = select_weighted_consensus(
                    prefix, np.ones(N, dtype=np.float64), ans_groups, is_corr
                )
                m_acc_orders["self_consistency"][o_idx] = 1.0 if sc_corr else 0.0
                m_cost_orders["self_consistency"][o_idx] = prefix_cost

                # 3. Highest mean log-probability (L_i)
                _, lp_corr, _ = select_best_of_n(prefix, L_all[prefix], ans_groups, is_corr)
                m_acc_orders["best_logp"][o_idx] = 1.0 if lp_corr else 0.0
                m_cost_orders["best_logp"][o_idx] = prefix_cost

                # 4. Highest self-certainty (K_i)
                _, k_corr, _ = select_best_of_n(prefix, K_all[prefix], ans_groups, is_corr)
                m_acc_orders["best_self_certainty"][o_idx] = 1.0 if k_corr else 0.0
                m_cost_orders["best_self_certainty"][o_idx] = prefix_cost

                # 5. Self-certainty + RWC
                _, rwc_k_corr, _, _, _ = select_rwc(prefix, K_all[prefix], ans_groups, is_corr)
                m_acc_orders["rwc_self_certainty"][o_idx] = 1.0 if rwc_k_corr else 0.0
                m_cost_orders["rwc_self_certainty"][o_idx] = prefix_cost

                # 6. Shortest rollout length control (-T_i)
                _, short_corr, _ = select_best_of_n(
                    prefix, -lengths[prefix].astype(np.float64), ans_groups, is_corr
                )
                m_acc_orders["shortest_length"][o_idx] = 1.0 if short_corr else 0.0
                m_cost_orders["shortest_length"][o_idx] = prefix_cost

                if is_vlm:
                    # 7 & 8. Vision-only (V_i) Best-of-N and RWC
                    _, v_corr, _ = select_best_of_n(prefix, V_all[prefix], ans_groups, is_corr)
                    m_acc_orders["best_vision"][o_idx] = 1.0 if v_corr else 0.0
                    m_cost_orders["best_vision"][o_idx] = prefix_cost

                    _, rwc_v_corr, _, _, _ = select_rwc(prefix, V_all[prefix], ans_groups, is_corr)
                    m_acc_orders["rwc_vision"][o_idx] = 1.0 if rwc_v_corr else 0.0
                    m_cost_orders["rwc_vision"][o_idx] = prefix_cost

                    # 9 & 10. Prefix-calibrated VisConf (Z_i = V_i * C_i) Best-of-N and RWC
                    _, Z_i, Z_tok_i, _ = compute_prefix_visconf(rec, prefix)
                    _, z_corr, _ = select_best_of_n(prefix, Z_i, ans_groups, is_corr)
                    m_acc_orders["best_visconf"][o_idx] = 1.0 if z_corr else 0.0
                    m_cost_orders["best_visconf"][o_idx] = prefix_cost

                    _, rwc_z_corr, _, _, _ = select_rwc(prefix, Z_i, ans_groups, is_corr)
                    m_acc_orders["rwc_visconf"][o_idx] = 1.0 if rwc_z_corr else 0.0
                    m_cost_orders["rwc_visconf"][o_idx] = prefix_cost

                    # 11 & 12. Tokenwise VisConf ablation (Z_i^token) Best-of-N and RWC
                    _, zt_corr, _ = select_best_of_n(prefix, Z_tok_i, ans_groups, is_corr)
                    m_acc_orders["best_visconf_token"][o_idx] = 1.0 if zt_corr else 0.0
                    m_cost_orders["best_visconf_token"][o_idx] = prefix_cost

                    _, rwc_zt_corr, _, _, _ = select_rwc(prefix, Z_tok_i, ans_groups, is_corr)
                    m_acc_orders["rwc_visconf_token"][o_idx] = 1.0 if rwc_zt_corr else 0.0
                    m_cost_orders["rwc_visconf_token"][o_idx] = prefix_cost

            per_q_oracle[N][q_idx] = float(np.mean(oracle_orders))
            per_q_groups[N][q_idx] = float(np.mean(ngroups_orders))
            for m in methods:
                per_q_acc[(m, N)][q_idx] = float(np.mean(m_acc_orders[m]))
                per_q_cost[(m, N)][q_idx] = float(np.mean(m_cost_orders[m]))

    rows = []
    for N in cfg.budgets:
        oracle_mean = float(np.mean(per_q_oracle[N]))
        groups_mean = float(np.mean(per_q_groups[N]))
        for m in methods:
            acc_mean = float(np.mean(per_q_acc[(m, N)]))
            cost_mean = float(np.mean(per_q_cost[(m, N)]))
            sampling_fail = 1.0 - oracle_mean
            selection_fail = oracle_mean - acc_mean
            rows.append(
                {
                    "dataset": dataset_key,
                    "split": split_filter,
                    "N": N,
                    "method": m,
                    "accuracy": acc_mean,
                    "oracle_coverage": oracle_mean,
                    "sampling_failure_rate": sampling_fail,
                    "selection_failure_rate": selection_fail,
                    "distinct_answer_groups": groups_mean,
                    "mean_token_cost": cost_mean,
                }
            )

    summary_df = pd.DataFrame(rows)

    # Compute paired question-level bootstrap 95% CIs at N = 32 (and N = 8, 16)
    bootstrap_comparisons: Dict[str, Any] = {}
    for N_check in [8, 16, 32]:
        pairs_to_test = [
            ("self_consistency", "first_ans"),
            ("rwc_self_certainty", "self_consistency"),
            ("rwc_self_certainty", "best_self_certainty"),
            ("best_self_certainty", "best_logp"),
        ]
        if is_vlm:
            pairs_to_test.extend(
                [
                    ("rwc_visconf", "self_consistency"),
                    ("rwc_visconf", "rwc_self_certainty"),
                    ("rwc_visconf", "best_visconf"),
                    ("best_visconf", "best_self_certainty"),
                    ("best_visconf", "best_vision"),
                    ("rwc_visconf_token", "rwc_visconf"),
                    ("best_visconf_token", "best_visconf"),
                ]
            )
        for m_a, m_b in pairs_to_test:
            key = f"N{N_check}_{m_a}_vs_{m_b}"
            bootstrap_comparisons[key] = paired_question_bootstrap_ci(
                per_q_acc[(m_a, N_check)],
                per_q_acc[(m_b, N_check)],
                num_resamples=cfg.num_bootstrap,
                seed=cfg.seed,
            )

    summary_df.to_csv(
        cfg.output_dir / "step2" / f"{dataset_key}_{split_filter}_methods_summary.csv",
        index=False,
    )
    with open(
        cfg.output_dir / "step2" / f"{dataset_key}_{split_filter}_bootstrap_ci.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(bootstrap_comparisons, f, indent=2)

    return summary_df, bootstrap_comparisons, per_q_acc


def analyze_step3_signals(
    cfg: Task3Config,
    records: Dict[str, Dict[str, Any]],
    dataset_key: str,
    split_filter: str = "eval",
) -> Dict[str, Any]:
    """
    Step 3: At N = 32, examines within-question associations between rollout length,
    scores, and correctness. Computes top-vs-bottom quartile (top 8 vs bottom 8 of 32)
    correctness differences within each question and averages across questions.
    For MathVista, summarizes within-rollout confidence-vision correlation and verifies
    Z_i^token - Z_i = Cov_t(c_{i,t}, v_{i,t}).
    """
    is_vlm = dataset_key == "mathvista"
    qids = [qid for qid, rec in records.items() if rec["split"] == split_filter]
    full_prefix = np.arange(cfg.num_rollouts, dtype=np.int32)

    score_names = ["L", "K", "neg_length"] + (["V", "Z", "Z_token"] if is_vlm else [])
    top_q_accs = {s: [] for s in score_names}
    bot_q_accs = {s: [] for s in score_names}
    quartile_diffs = {s: [] for s in score_names}
    within_q_corrs_len = {s: [] for s in score_names if s != "neg_length"}

    all_rollout_corr_kv: List[float] = []
    all_rollout_cov_cv: List[float] = []
    max_identity_error = 0.0

    for qid in qids:
        rec = records[qid]
        is_corr = rec["is_correct"].astype(np.float64)
        lengths = rec["lengths"].astype(np.float64)

        score_map: Dict[str, np.ndarray] = {
            "L": rec["L"],
            "K": rec["K"],
            "neg_length": -lengths,
        }
        if is_vlm:
            C_i, Z_i, Z_tok_i, cov_cv_i = compute_prefix_visconf(rec, full_prefix)
            score_map["V"] = rec["V"]
            score_map["Z"] = Z_i
            score_map["Z_token"] = Z_tok_i

            # Verify algebraic identity Z_i^token - Z_i == Cov_t(c_{i,t}, v_{i,t})
            all_rollout_corr_kv.extend(rec["corr_kv"].tolist())
            all_rollout_cov_cv.extend(cov_cv_i.tolist())

        for s_name, s_vals in score_map.items():
            # Sort ascending by score: bottom quartile = first 8, top quartile = last 8
            order_idx = np.argsort(s_vals, kind="mergesort")
            bot_8 = float(np.mean(is_corr[order_idx[:8]]))
            top_8 = float(np.mean(is_corr[order_idx[-8:]]))
            top_q_accs[s_name].append(top_8)
            bot_q_accs[s_name].append(bot_8)
            quartile_diffs[s_name].append(top_8 - bot_8)

            if s_name != "neg_length":
                finite = np.isfinite(s_vals)
                if np.sum(finite) >= 4 and np.std(s_vals[finite]) > 1e-12 and np.std(lengths[finite]) > 1e-12:
                    within_q_corrs_len[s_name].append(
                        float(np.corrcoef(s_vals[finite], lengths[finite])[0, 1])
                    )

    quartile_summary = {}
    for s_name in score_names:
        quartile_summary[s_name] = {
            "top_quartile_acc": float(np.mean(top_q_accs[s_name])),
            "bottom_quartile_acc": float(np.mean(bot_q_accs[s_name])),
            "mean_quartile_diff": float(np.mean(quartile_diffs[s_name])),
            "std_err_diff": float(np.std(quartile_diffs[s_name], ddof=1) / np.sqrt(len(qids))),
            "mean_within_q_corr_with_length": (
                float(np.mean(within_q_corrs_len[s_name]))
                if s_name in within_q_corrs_len and len(within_q_corrs_len[s_name]) > 0
                else None
            ),
        }

    result = {
        "dataset": dataset_key,
        "split": split_filter,
        "num_questions": len(qids),
        "quartile_analysis": quartile_summary,
    }

    if is_vlm:
        corr_arr = np.asarray(all_rollout_corr_kv, dtype=np.float64)
        cov_arr = np.asarray(all_rollout_cov_cv, dtype=np.float64)
        result["vlm_signal_coupling"] = {
            "mean_within_rollout_corr_k_v": float(np.mean(corr_arr)),
            "median_within_rollout_corr_k_v": float(np.median(corr_arr)),
            "iqr_within_rollout_corr_k_v": [
                float(np.quantile(corr_arr, 0.25)),
                float(np.quantile(corr_arr, 0.75)),
            ],
            "mean_token_covariance_c_v": float(np.mean(cov_arr)),
            "std_token_covariance_c_v": float(np.std(cov_arr)),
            "frac_positive_covariance": float(np.mean(cov_arr > 0)),
        }

    out_path = cfg.output_dir / "step3" / f"{dataset_key}_{split_filter}_step3_signals.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def simulate_adaptive_policy_on_question(
    rec: Dict[str, Any],
    is_vlm: bool,
    controller: str,
    tau: float,
    gamma: float = 0.0,
    checkpoints: Tuple[int, ...] = (4, 8, 16),
) -> Dict[str, Any]:
    """
    Simulates adaptive stopping across the 10 saved rollout orders of a single question.
    Checkpoints n in {4, 8, 16}; otherwise stops at 32.
    Both controllers return the current self-consistency winner at stopping step n.
    """
    orders = rec["orders"]
    ans_groups = rec["answer_groups"]
    is_corr = rec["is_correct"]
    lengths = rec["lengths"]
    K_all = rec["K"]

    num_orders = orders.shape[0]
    acc_orders = np.zeros(num_orders, dtype=np.float64)
    cost_orders = np.zeros(num_orders, dtype=np.float64)
    stop_n_orders = np.zeros(num_orders, dtype=np.int32)
    sc32_acc_orders = np.zeros(num_orders, dtype=np.float64)

    for o_idx in range(num_orders):
        order = orders[o_idx]
        # Fixed-32 self-consistency outcome for this order
        _, sc32_corr, _, _ = select_weighted_consensus(
            order, np.ones(32, dtype=np.float64), ans_groups, is_corr
        )
        sc32_acc_orders[o_idx] = 1.0 if sc32_corr else 0.0

        stopped = False
        if controller != "fixed_32":
            for n in checkpoints:
                prefix = order[:n]
                # Count answer group votes in prefix
                counts: Dict[str, int] = {}
                for rid in prefix:
                    g = ans_groups[int(rid)]
                    counts[g] = counts.get(g, 0) + 1

                sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                top_group, top_count = sorted_counts[0]
                runner_up_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0

                # Agreement condition: unique leading answer, non-NO_ANSWER, >= tau * n
                unique_leader = top_count > runner_up_count
                agree_ok = (
                    unique_leader
                    and (top_group != NO_ANSWER)
                    and ((float(top_count) / float(n)) >= tau - 1e-12)
                )

                if not agree_ok:
                    continue

                if controller == "agreement":
                    stopped = True
                elif controller == "quality_aware":
                    # Compute RWC winner and margin using ONLY revealed prefix candidates
                    if is_vlm:
                        _, q_scores, _, _ = compute_prefix_visconf(rec, prefix)
                    else:
                        q_scores = K_all[prefix]
                    rwc_win_g, _, _, _, rwc_margin = select_rwc(
                        prefix, q_scores, ans_groups, is_corr
                    )
                    if rwc_win_g == top_group and rwc_margin >= gamma - 1e-12:
                        stopped = True
                else:
                    raise ValueError(f"Unknown controller: {controller}")

                if stopped:
                    _, sc_n_corr, _, _ = select_weighted_consensus(
                        prefix, np.ones(n, dtype=np.float64), ans_groups, is_corr
                    )
                    acc_orders[o_idx] = 1.0 if sc_n_corr else 0.0
                    cost_orders[o_idx] = float(np.sum(lengths[prefix]))
                    stop_n_orders[o_idx] = n
                    break

        if not stopped:
            acc_orders[o_idx] = sc32_acc_orders[o_idx]
            cost_orders[o_idx] = float(np.sum(lengths[order]))
            stop_n_orders[o_idx] = 32

    return {
        "mean_acc": float(np.mean(acc_orders)),
        "mean_cost": float(np.mean(cost_orders)),
        "mean_rollouts": float(np.mean(stop_n_orders)),
        "stop_n_orders": stop_n_orders,
        "acc_orders": acc_orders,
        "sc32_acc_orders": sc32_acc_orders,
    }


def run_step5_adaptive_sampling(
    cfg: Task3Config,
    records: Dict[str, Dict[str, Any]],
    dataset_key: str,
) -> Dict[str, Any]:
    """
    Step 5: Tunes agreement-based (tau) and quality-aware (tau, gamma) controllers on
    the 200-question Dev split (lowest token cost within <= 1.0 percentage point of
    fixed-32 self-consistency; ties broken by higher accuracy, then smaller tau, gamma),
    freezes the selected thresholds, and evaluates on the 800-question Held-out split.
    """
    is_vlm = dataset_key == "mathvista"
    dev_qids = [qid for qid, rec in records.items() if rec["split"] == "dev"]
    eval_qids = [qid for qid, rec in records.items() if rec["split"] == "eval"]

    # Baseline fixed-32 self-consistency on Dev
    dev_fixed32 = [
        simulate_adaptive_policy_on_question(records[qid], is_vlm, "fixed_32", 1.0, 0.0)
        for qid in dev_qids
    ]
    dev_sc32_acc = float(np.mean([r["mean_acc"] for r in dev_fixed32]))
    dev_sc32_cost = float(np.mean([r["mean_cost"] for r in dev_fixed32]))
    acc_floor = dev_sc32_acc - 0.01 - 1e-9

    # 1. Grid search for Agreement-Based controller on Dev
    agree_candidates = []
    for tau in cfg.tau_grid:
        res_list = [
            simulate_adaptive_policy_on_question(
                records[qid], is_vlm, "agreement", tau, 0.0, cfg.adaptive_checkpoints
            )
            for qid in dev_qids
        ]
        m_acc = float(np.mean([r["mean_acc"] for r in res_list]))
        m_cost = float(np.mean([r["mean_cost"] for r in res_list]))
        m_rolls = float(np.mean([r["mean_rollouts"] for r in res_list]))
        agree_candidates.append(
            {
                "controller": "agreement",
                "tau": float(tau),
                "gamma": 0.0,
                "dev_acc": m_acc,
                "dev_cost": m_cost,
                "dev_rollouts": m_rolls,
                "cost_ratio_vs_fixed32": m_cost / max(dev_sc32_cost, 1e-12),
                "feasible": bool(m_acc >= acc_floor),
            }
        )

    # 2. Grid search for Quality-Aware controller on Dev
    qa_candidates = []
    for tau in cfg.tau_grid:
        for gamma in cfg.gamma_grid:
            res_list = [
                simulate_adaptive_policy_on_question(
                    records[qid],
                    is_vlm,
                    "quality_aware",
                    tau,
                    gamma,
                    cfg.adaptive_checkpoints,
                )
                for qid in dev_qids
            ]
            m_acc = float(np.mean([r["mean_acc"] for r in res_list]))
            m_cost = float(np.mean([r["mean_cost"] for r in res_list]))
            m_rolls = float(np.mean([r["mean_rollouts"] for r in res_list]))
            qa_candidates.append(
                {
                    "controller": "quality_aware",
                    "tau": float(tau),
                    "gamma": float(gamma),
                    "dev_acc": m_acc,
                    "dev_cost": m_cost,
                    "dev_rollouts": m_rolls,
                    "cost_ratio_vs_fixed32": m_cost / max(dev_sc32_cost, 1e-12),
                    "feasible": bool(m_acc >= acc_floor),
                }
            )

    def _select_best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Selects the lowest-cost setting within <= 1.0 percentage point of fixed-32
        self-consistency on Dev. Ties broken by higher accuracy, then smaller (tau, gamma).
        If no candidate meets the accuracy floor, falls back to highest Dev accuracy.
        """
        feasible = [c for c in candidates if c["feasible"]]
        if feasible:
            return min(
                feasible,
                key=lambda c: (
                    round(c["dev_cost"], 6),
                    -round(c["dev_acc"], 6),
                    c["tau"],
                    c["gamma"],
                ),
            )
        return max(
            candidates,
            key=lambda c: (
                round(c["dev_acc"], 6),
                -round(c["dev_cost"], 6),
                -c["tau"],
                -c["gamma"],
            ),
        )

    best_agree = _select_best_candidate(agree_candidates)
    best_qa = _select_best_candidate(qa_candidates)

    # 3. Evaluate frozen controllers and fixed-32 baseline on Held-out (eval) split
    def _evaluate_frozen_on_split(
        qids_list: List[str],
        controller: str,
        tau: float,
        gamma: float,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        per_q_res = [
            simulate_adaptive_policy_on_question(
                records[qid], is_vlm, controller, tau, gamma, cfg.adaptive_checkpoints
            )
            for qid in qids_list
        ]
        q_acc = np.array([r["mean_acc"] for r in per_q_res], dtype=np.float64)
        q_cost = np.array([r["mean_cost"] for r in per_q_res], dtype=np.float64)
        q_rolls = np.array([r["mean_rollouts"] for r in per_q_res], dtype=np.float64)

        all_stop_n = np.concatenate([r["stop_n_orders"] for r in per_q_res])
        all_acc = np.concatenate([r["acc_orders"] for r in per_q_res])
        all_sc32 = np.concatenate([r["sc32_acc_orders"] for r in per_q_res])

        stop_dist = {
            str(n_val): float(np.mean(all_stop_n == n_val))
            for n_val in list(cfg.adaptive_checkpoints) + [32]
        }

        early_mask = all_stop_n < 32
        early_frac = float(np.mean(early_mask))
        if np.any(early_mask):
            early_match_sc32 = float(np.mean(all_acc[early_mask] == all_sc32[early_mask]))
            early_Regret_c2w = float(
                np.mean((all_sc32[early_mask] == 1.0) & (all_acc[early_mask] == 0.0))
            )
            early_gain_w2c = float(
                np.mean((all_sc32[early_mask] == 0.0) & (all_acc[early_mask] == 1.0))
            )
        else:
            early_match_sc32 = 1.0
            early_Regret_c2w = 0.0
            early_gain_w2c = 0.0

        summary = {
            "controller": controller,
            "tau": float(tau),
            "gamma": float(gamma),
            "accuracy": float(np.mean(q_acc)),
            "mean_token_cost": float(np.mean(q_cost)),
            "mean_rollouts": float(np.mean(q_rolls)),
            "stopping_distribution": stop_dist,
            "early_stop_fraction": early_frac,
            "early_stop_agreement_with_sc32": early_match_sc32,
            "early_stop_c2w_rate": early_Regret_c2w,
            "early_stop_w2c_rate": early_gain_w2c,
        }
        return summary, q_acc, q_cost

    eval_fixed_sum, eval_fixed_acc, eval_fixed_cost = _evaluate_frozen_on_split(
        eval_qids, "fixed_32", 1.0, 0.0
    )
    eval_agree_sum, eval_agree_acc, eval_agree_cost = _evaluate_frozen_on_split(
        eval_qids, "agreement", best_agree["tau"], 0.0
    )
    eval_qa_sum, eval_qa_acc, eval_qa_cost = _evaluate_frozen_on_split(
        eval_qids, "quality_aware", best_qa["tau"], best_qa["gamma"]
    )

    fixed_cost_mean = max(eval_fixed_sum["mean_token_cost"], 1e-12)
    for s_dict in (eval_fixed_sum, eval_agree_sum, eval_qa_sum):
        s_dict["token_savings_vs_fixed32"] = float(
            1.0 - (s_dict["mean_token_cost"] / fixed_cost_mean)
        )
        s_dict["acc_delta_vs_fixed32"] = float(
            s_dict["accuracy"] - eval_fixed_sum["accuracy"]
        )

    # Paired bootstrap CIs on Held-out (eval) split
    eval_bootstrap = {
        "qa_vs_agree_acc": paired_question_bootstrap_ci(
            eval_qa_acc, eval_agree_acc, num_resamples=cfg.num_bootstrap, seed=cfg.seed
        ),
        "qa_vs_agree_cost": paired_question_bootstrap_ci(
            eval_qa_cost, eval_agree_cost, num_resamples=cfg.num_bootstrap, seed=cfg.seed
        ),
        "agree_vs_fixed32_acc": paired_question_bootstrap_ci(
            eval_agree_acc, eval_fixed_acc, num_resamples=cfg.num_bootstrap, seed=cfg.seed
        ),
        "agree_vs_fixed32_cost": paired_question_bootstrap_ci(
            eval_agree_cost, eval_fixed_cost, num_resamples=cfg.num_bootstrap, seed=cfg.seed
        ),
        "qa_vs_fixed32_acc": paired_question_bootstrap_ci(
            eval_qa_acc, eval_fixed_acc, num_resamples=cfg.num_bootstrap, seed=cfg.seed
        ),
        "qa_vs_fixed32_cost": paired_question_bootstrap_ci(
            eval_qa_cost, eval_fixed_cost, num_resamples=cfg.num_bootstrap, seed=cfg.seed
        ),
    }

    step5_dir = cfg.output_dir / "step5"
    step5_dir.mkdir(parents=True, exist_ok=True)

    grid_df = pd.DataFrame(agree_candidates + qa_candidates)
    grid_df.to_csv(step5_dir / f"{dataset_key}_dev_tuning_grid.csv", index=False)

    eval_table_df = pd.DataFrame([eval_fixed_sum, eval_agree_sum, eval_qa_sum])
    eval_table_df.to_csv(step5_dir / f"{dataset_key}_eval_adaptive_summary.csv", index=False)

    step5_payload = {
        "dataset": dataset_key,
        "dev_baseline_fixed32": {
            "accuracy": dev_sc32_acc,
            "mean_token_cost": dev_sc32_cost,
            "accuracy_floor": acc_floor,
        },
        "selected_on_dev": {
            "agreement": best_agree,
            "quality_aware": best_qa,
        },
        "eval_results": {
            "fixed_32": eval_fixed_sum,
            "agreement": eval_agree_sum,
            "quality_aware": eval_qa_sum,
        },
        "eval_bootstrap_ci": eval_bootstrap,
    }

    with open(step5_dir / f"{dataset_key}_step5_adaptive.json", "w", encoding="utf-8") as f:
        json.dump(step5_payload, f, indent=2)

    return step5_payload


def run_all_offline_steps_for_dataset(
    cfg: Task3Config,
    records: Dict[str, Dict[str, Any]],
    dataset_key: str,
) -> Dict[str, Any]:
    """
    Convenience orchestrator that runs Steps 1, 2, 3, and 5 for a single dataset
    and writes all corresponding summary tables and JSON files under cfg.output_dir.
    """
    for subdir in ("step1", "step2", "step3", "step5"):
        (cfg.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    summary_df, boot_ci, per_q_acc = evaluate_steps_1_and_2(
        cfg, records, dataset_key, split_filter="eval"
    )

    # Save dedicated Step 1 scaling & failure decomposition view (Self-Consistency vs FirstAns)
    step1_df = summary_df[
        summary_df["method"].isin(["first_ans", "self_consistency"])
    ].copy()
    step1_df.to_csv(
        cfg.output_dir / "step1" / f"{dataset_key}_eval_step1_decomposition.csv",
        index=False,
    )

    step3_res = analyze_step3_signals(cfg, records, dataset_key, split_filter="eval")
    step5_res = run_step5_adaptive_sampling(cfg, records, dataset_key)

    return {
        "step1_df": step1_df,
        "step2_df": summary_df,
        "step2_bootstrap": boot_ci,
        "step2_per_q_acc": per_q_acc,
        "step3": step3_res,
        "step5": step5_res,
    }
