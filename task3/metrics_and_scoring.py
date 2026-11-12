from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata

from .answer_utils import NO_ANSWER


def compute_prefix_visconf(
    q_rec: Dict[str, Any], prefix_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes candidate-prefix-calibrated confidence C_i, trajectory VisConf Z_i = V_i * C_i,
    tokenwise fusion Z_i^token = (1/T_i) sum_t c_{i,t} * v_{i,t}, and token covariance
    Cov_t(c_{i,t}, v_{i,t}) for the candidate subset `prefix_indices` (size N).
    Uses ONLY tokens from `prefix_indices` and linear-interpolation percentiles.
    """
    N = len(prefix_indices)
    offsets = q_rec["offsets"]
    kl_flat = q_rec["kl_flat"]
    v_flat = q_rec["v_flat"]
    V_sub = q_rec["V"][prefix_indices]

    # Collect slices for candidates in prefix_indices
    starts = offsets[prefix_indices]
    ends = offsets[prefix_indices + 1]
    lengths = ends - starts
    total_tokens = int(np.sum(lengths))

    if total_tokens == 0:
        zeros = np.zeros(N, dtype=np.float64)
        return zeros, zeros, zeros, zeros

    # Pool token KL values across G = prefix_indices
    k_chunks = [kl_flat[s:e] for s, e in zip(starts, ends) if e > s]
    v_chunks = [v_flat[s:e] for s, e in zip(starts, ends) if e > s]
    k_pool = np.concatenate(k_chunks)
    v_pool = np.concatenate(v_chunks)

    q05, q95 = np.quantile(k_pool, [0.05, 0.95], method="linear")
    denom = max(float(q95 - q05), 1e-8)
    c_pool = np.clip((k_pool - float(q05)) / denom, 0.0, 1.0)
    cv_pool = c_pool * v_pool

    # Fast reduction per rollout using np.add.reduceat
    C_i = np.zeros(N, dtype=np.float64)
    Z_token_i = np.zeros(N, dtype=np.float64)

    non_empty_mask = lengths > 0
    non_empty_lengths = lengths[non_empty_mask]
    reduce_starts = np.zeros(len(non_empty_lengths), dtype=np.int64)
    if len(non_empty_lengths) > 1:
        reduce_starts[1:] = np.cumsum(non_empty_lengths[:-1])

    c_sums = np.add.reduceat(c_pool.astype(np.float64), reduce_starts)
    cv_sums = np.add.reduceat(cv_pool.astype(np.float64), reduce_starts)

    C_i[non_empty_mask] = c_sums / non_empty_lengths
    Z_token_i[non_empty_mask] = cv_sums / non_empty_lengths
    Z_i = V_sub * C_i
    cov_cv_i = Z_token_i - Z_i

    return C_i, Z_i, Z_token_i, cov_cv_i


def select_best_of_n(
    prefix_indices: np.ndarray,
    scores: np.ndarray,
    answer_groups: List[str],
    is_correct: np.ndarray,
) -> Tuple[str, bool, int]:
    """
    Highest-score selection among candidates in `prefix_indices`.
    Ties in `scores` are resolved by the earliest rollout in `prefix_indices`
    (guaranteed by np.argmax on a 1D array).
    Returns (selected_group, selected_is_correct, winning_rollout_id).
    """
    best_pos = int(np.argmax(scores))
    win_rid = int(prefix_indices[best_pos])
    return answer_groups[win_rid], bool(is_correct[win_rid]), win_rid


def select_weighted_consensus(
    prefix_indices: np.ndarray,
    weights: np.ndarray,
    answer_groups: List[str],
    is_correct: np.ndarray,
) -> Tuple[str, bool, int, Dict[str, float]]:
    """
    Aggregates candidate votes `weights` by answer group.
    Ties in total group vote W(a) are resolved by the earliest supporting rollout
    in the current order (`prefix_indices`).
    Returns (winning_group, is_correct, earliest_supporting_rollout_id, group_weights_dict).
    """
    group_weights: Dict[str, float] = {}
    group_first_pos: Dict[str, int] = {}
    group_first_rid: Dict[str, int] = {}
    group_is_corr: Dict[str, bool] = {}

    for pos, (rid_raw, w) in enumerate(zip(prefix_indices, weights)):
        rid = int(rid_raw)
        g = answer_groups[rid]
        if g not in group_weights:
            group_weights[g] = 0.0
            group_first_pos[g] = pos
            group_first_rid[g] = rid
            group_is_corr[g] = bool(is_correct[rid])
        group_weights[g] += float(w)

    # Sort groups by (-W(a), earliest_pos) with small float tolerance 1e-12 for ties
    best_group = None
    best_weight = -np.inf
    best_pos = len(prefix_indices) + 1

    for g, w in group_weights.items():
        first_p = group_first_pos[g]
        if w > best_weight + 1e-12 or (abs(w - best_weight) <= 1e-12 and first_p < best_pos):
            best_group = g
            best_weight = w
            best_pos = first_p

    win_rid = group_first_rid[best_group]
    return best_group, group_is_corr[best_group], win_rid, group_weights


def compute_rwc_weights(scores: np.ndarray) -> np.ndarray:
    """
    Computes Rank-Weighted Consensus (RWC) weights for N candidates:
      - At N == 1, w_1 = 1.0
      - At N > 1, descending average rank r_i in [1, N] (rank 1 = highest score),
        w_i = (N - r_i) / (N - 1).
    """
    N = len(scores)
    if N == 1:
        return np.ones(1, dtype=np.float64)
    # Replace -inf with a finite minimum below all valid scores so rankdata handles ties cleanly
    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        clean_scores = np.zeros(N, dtype=np.float64)
    else:
        min_val = float(np.min(scores[finite_mask]))
        clean_scores = np.where(finite_mask, scores, min_val - 1e6)

    ranks = rankdata(-clean_scores, method="average")
    return (N - ranks) / float(N - 1)


def select_rwc(
    prefix_indices: np.ndarray,
    scores: np.ndarray,
    answer_groups: List[str],
    is_correct: np.ndarray,
) -> Tuple[str, bool, int, Dict[str, float], float]:
    """
    Runs Rank-Weighted Consensus (RWC) on `prefix_indices` using `scores`.
    Also computes the RWC vote margin: (W_(1) - W_(2)) / sum_a W(a),
    using 0 for W_(2) when only one answer group exists.
    """
    if len(prefix_indices) == 1:
        rid = int(prefix_indices[0])
        g = answer_groups[rid]
        return g, bool(is_correct[rid]), rid, {g: 1.0}, 1.0

    weights = compute_rwc_weights(scores)
    win_g, win_corr, win_rid, group_weights = select_weighted_consensus(
        prefix_indices, weights, answer_groups, is_correct
    )
    vals = sorted(group_weights.values(), reverse=True)
    total_w = sum(vals)
    if total_w <= 1e-12:
        margin = 0.0
    elif len(vals) == 1:
        margin = 1.0
    else:
        margin = float((vals[0] - vals[1]) / total_w)

    return win_g, win_corr, win_rid, group_weights, margin


def paired_question_bootstrap_ci(
    acc_a_per_q: np.ndarray,
    acc_b_per_q: np.ndarray,
    num_resamples: int = 1000,
    seed: int = 6304,
) -> Dict[str, float]:
    """
    Computes 1,000 paired question-level bootstrap resamples for 95% confidence
    intervals on accuracy difference (Method A - Method B), after averaging over orders.
    """
    diff = np.asarray(acc_a_per_q, dtype=np.float64) - np.asarray(acc_b_per_q, dtype=np.float64)
    n_q = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_q, size=(num_resamples, n_q))
    boot_means = np.mean(diff[idx], axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975], method="linear")
    return {
        "mean_diff": float(np.mean(diff)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "std_err": float(np.std(boot_means, ddof=1)),
    }
