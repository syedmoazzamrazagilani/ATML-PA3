import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .answer_utils import (
    HAS_MATH_VERIFY,
    MATH_VERIFY_VERSION,
    NO_ANSWER,
    group_and_grade_math_question,
    group_and_grade_mathvista_question,
)
from .config import Task3Config


def load_and_join_dataset(
    cfg: Task3Config, dataset_key: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads and joins the parquet tables for 'math' (qwen25-math) or 'mathvista' (qwen25vl-mathvista).
    Verifies row counts, join keys, and per-token array alignment.
    """
    if dataset_key == "math":
        folder = cfg.data_root / "qwen25-math"
        is_vlm = False
    elif dataset_key == "mathvista":
        folder = cfg.data_root / "qwen25vl-mathvista"
        is_vlm = True
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")

    gen_path = folder / "generations.parquet"
    prob_path = folder / "probability_metrics.parquet"
    if not gen_path.exists() or not prob_path.exists():
        raise FileNotFoundError(f"Missing parquet files in {folder}")

    join_keys = ["dataset", "sample_id", "rollout_id"]
    df_gen = pd.read_parquet(gen_path)
    df_prob = pd.read_parquet(prob_path)

    df = pd.merge(df_gen, df_prob, on=join_keys, how="inner", validate="one_to_one")

    if is_vlm:
        attn_path = folder / "attention_metrics.parquet"
        df_attn = pd.read_parquet(attn_path)
        # Rename attention_metrics's per-token list n_generated_text_tokens to prevent collision
        if "n_generated_text_tokens" in df_attn.columns:
            df_attn = df_attn.rename(
                columns={"n_generated_text_tokens": "preceding_generated_tokens_list"}
            )
        df = pd.merge(df, df_attn, on=join_keys, how="inner", validate="one_to_one")

    df["sample_id"] = df["sample_id"].astype(str)
    df["rollout_id"] = df["rollout_id"].astype(int)
    df = df.sort_values(["sample_id", "rollout_id"]).reset_index(drop=True)

    # Sanity checks against README specifications
    n_rows = len(df)
    n_questions = df["sample_id"].nunique()
    total_tokens = int(df["n_generated_text_tokens"].sum())
    capped_rollouts = int(df["hit_max_new_tokens"].sum())
    null_extracted = int(df["extracted_answer"].isna().sum())

    # Verify token alignment across all rows
    len_tok = df["generated_token_ids"].map(len)
    len_logp = df["logp"].map(len)
    len_kl = df["kl_u_p"].map(len)
    aligned = (
        (len_tok == df["n_generated_text_tokens"]).all()
        and (len_logp == df["n_generated_text_tokens"]).all()
        and (len_kl == df["n_generated_text_tokens"]).all()
    )
    if is_vlm:
        len_img = df["img_attn_total"].map(len)
        len_prm = df["prompt_text_attn_total"].map(len)
        len_gen = df["generated_text_attn_total"].map(len)
        aligned = (
            aligned
            and (len_img == df["n_generated_text_tokens"]).all()
            and (len_prm == df["n_generated_text_tokens"]).all()
            and (len_gen == df["n_generated_text_tokens"]).all()
        )
    if not aligned:
        raise AssertionError(f"Token alignment check failed for {dataset_key}!")

    stats = {
        "dataset_key": dataset_key,
        "n_rows": n_rows,
        "n_questions": n_questions,
        "total_retained_tokens": total_tokens,
        "capped_rollouts": capped_rollouts,
        "null_extracted_answers": null_extracted,
        "token_alignment_verified": bool(aligned),
        "math_verify_version": MATH_VERIFY_VERSION,
        "has_math_verify": HAS_MATH_VERIFY,
    }
    return df, stats


def create_and_save_splits(
    cfg: Task3Config, sample_ids: List[str], dataset_key: str
) -> Dict[str, Any]:
    """
    Sorts the 1,000 question IDs lexicographically and permutes them with
    np.random.default_rng(6304): first 200 Dev, remaining 800 Eval.
    Also generates 10 seeded random permutations of the 32 rollout IDs per question.
    """
    sorted_ids = sorted(set(str(x) for x in sample_ids))
    assert len(sorted_ids) == cfg.num_dev + cfg.num_eval, (
        f"Expected {cfg.num_dev + cfg.num_eval} unique sample_ids, got {len(sorted_ids)}"
    )

    rng = np.random.default_rng(cfg.seed)
    perm_indices = rng.permutation(len(sorted_ids))
    permuted_ids = [sorted_ids[i] for i in perm_indices]

    dev_ids = permuted_ids[: cfg.num_dev]
    eval_ids = permuted_ids[cfg.num_dev :]
    live_50_ids = eval_ids[: cfg.num_live_math] if dataset_key == "math" else []

    # Generate 10 random permutations of rollout IDs 0..31 for each question (keyed by sorted_ids)
    order_rng = np.random.default_rng(cfg.seed)
    orders_by_qid: Dict[str, List[List[int]]] = {}
    for qid in sorted_ids:
        q_orders = [
            order_rng.permutation(cfg.num_rollouts).astype(int).tolist()
            for _ in range(cfg.num_orders)
        ]
        orders_by_qid[qid] = q_orders

    split_info = {
        "dataset_key": dataset_key,
        "seed": cfg.seed,
        "dev_ids": dev_ids,
        "eval_ids": eval_ids,
        "live_50_ids": live_50_ids,
        "orders_by_qid": orders_by_qid,
    }

    out_path = cfg.output_dir / "splits" / f"{dataset_key}_splits_and_orders.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    return split_info


def build_compact_question_records(
    cfg: Task3Config,
    df: pd.DataFrame,
    dataset_key: str,
    split_info: Dict[str, Any],
    mathvista_meta: Dict[str, Any] = None,
    use_cache: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Builds a memory-efficient dictionary of per-question data structures containing:
      - rollout lengths T_i (int32[32])
      - hit_max_new_tokens (bool[32])
      - raw extracted_answer, normalized answer_group (List[str] of len 32), is_correct (bool[32])
      - scalar rollout scores L_i (mean logp), K_i (mean kl_u_p), and V_i (mean visual engagement)
      - within-rollout correlation Corr_t(k_{i,t}, v_{i,t})
      - contiguous flat float32 arrays kl_flat (and v_flat for MathVista) + offsets int32[33]
        so candidate-prefix VisConf calibration runs in microseconds per question.
    """
    cache_path = cfg.output_dir / "cache" / f"{dataset_key}_compact_records.npz"
    meta_cache_path = cfg.output_dir / "cache" / f"{dataset_key}_question_meta.json"

    is_vlm = dataset_key == "mathvista"
    dev_set = set(split_info["dev_ids"])

    records: Dict[str, Dict[str, Any]] = {}
    grouped = df.groupby("sample_id", sort=True)

    for qid, g in tqdm(grouped, total=len(grouped), desc=f"Preprocessing {dataset_key}"):
        g = g.sort_values("rollout_id")
        assert len(g) == cfg.num_rollouts, f"Question {qid} has {len(g)} rollouts != 32"

        extracted_list = [
            None if pd.isna(x) else str(x) for x in g["extracted_answer"].tolist()
        ]
        gold_ans = str(g["gold_answer"].iloc[0])
        q_text = str(g["question_text"].iloc[0])

        if dataset_key == "math":
            ans_groups, is_corr = group_and_grade_math_question(extracted_list, gold_ans)
        else:
            q_meta = (mathvista_meta or {}).get(str(qid), {})
            ans_groups, is_corr = group_and_grade_mathvista_question(
                extracted_list, gold_ans, q_meta
            )

        lengths = g["n_generated_text_tokens"].to_numpy(dtype=np.int32)
        capped = g["hit_max_new_tokens"].to_numpy(dtype=bool)

        L_arr = np.zeros(cfg.num_rollouts, dtype=np.float64)
        K_arr = np.zeros(cfg.num_rollouts, dtype=np.float64)
        V_arr = np.zeros(cfg.num_rollouts, dtype=np.float64)
        corr_kv_arr = np.zeros(cfg.num_rollouts, dtype=np.float64)

        kl_lists = g["kl_u_p"].tolist()
        logp_lists = g["logp"].tolist()

        offsets = np.zeros(cfg.num_rollouts + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(lengths)
        total_q_tokens = int(offsets[-1])

        kl_flat = np.empty(total_q_tokens, dtype=np.float32)
        v_flat = np.empty(total_q_tokens, dtype=np.float32) if is_vlm else None

        if is_vlm:
            img_lists = g["img_attn_total"].tolist()
            prm_lists = g["prompt_text_attn_total"].tolist()
            gen_lists = g["generated_text_attn_total"].tolist()

        for r_idx in range(cfg.num_rollouts):
            s_idx, e_idx = int(offsets[r_idx]), int(offsets[r_idx + 1])
            T_i = e_idx - s_idx
            if T_i <= 0:
                L_arr[r_idx] = -np.inf
                K_arr[r_idx] = -np.inf
                V_arr[r_idx] = 0.0
                corr_kv_arr[r_idx] = 0.0
                continue

            lp = np.asarray(logp_lists[r_idx], dtype=np.float32)
            kl = np.asarray(kl_lists[r_idx], dtype=np.float32)
            kl_flat[s_idx:e_idx] = kl
            L_arr[r_idx] = float(np.mean(lp, dtype=np.float64))
            K_arr[r_idx] = float(np.mean(kl, dtype=np.float64))

            if is_vlm:
                a_img = np.asarray(img_lists[r_idx], dtype=np.float32)
                a_prm = np.asarray(prm_lists[r_idx], dtype=np.float32)
                a_gen = np.asarray(gen_lists[r_idx], dtype=np.float32)
                denom = np.maximum(a_img + a_prm + a_gen, 1e-12)
                v_t = a_img / denom
                v_flat[s_idx:e_idx] = v_t
                V_arr[r_idx] = float(np.mean(v_t, dtype=np.float64))

                if T_i >= 2 and np.std(kl) > 1e-12 and np.std(v_t) > 1e-12:
                    corr_kv_arr[r_idx] = float(np.corrcoef(kl, v_t)[0, 1])
                else:
                    corr_kv_arr[r_idx] = 0.0

        records[str(qid)] = {
            "sample_id": str(qid),
            "split": "dev" if str(qid) in dev_set else "eval",
            "question_text": q_text,
            "gold_answer": gold_ans,
            "extracted_answers": extracted_list,
            "answer_groups": ans_groups,
            "is_correct": np.asarray(is_corr, dtype=bool),
            "lengths": lengths,
            "hit_max_new_tokens": capped,
            "L": L_arr,
            "K": K_arr,
            "V": V_arr,
            "corr_kv": corr_kv_arr,
            "offsets": offsets,
            "kl_flat": kl_flat,
            "v_flat": v_flat,
            "orders": np.asarray(split_info["orders_by_qid"][str(qid)], dtype=np.int32),
        }

    return records
