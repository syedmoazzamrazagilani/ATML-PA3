import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

try:
    from . import answer_utils
    from .answer_utils import NO_ANSWER
    from .config import Task3Config
    from .metrics_and_scoring import (
        compute_prefix_visconf,
        paired_question_bootstrap_ci,
        select_best_of_n,
        select_rwc,
        select_weighted_consensus,
    )
except ImportError:
    import answer_utils
    from answer_utils import NO_ANSWER
    from config import Task3Config
    from metrics_and_scoring import (
        compute_prefix_visconf,
        paired_question_bootstrap_ci,
        select_best_of_n,
        select_rwc,
        select_weighted_consensus,
    )


def _get_math_verify_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("math-verify")
    except Exception:
        return "unknown"


def extract_last_boxed_balanced(text: str) -> str:
    """
    Extracts the last \\boxed{...} or \\fbox{...} expression using balanced braces.
    Returns NO_ANSWER if no balanced boxed expression is found.
    """
    if hasattr(answer_utils, "extract_boxed_answer"):
        res = answer_utils.extract_boxed_answer(text)
        return str(res).strip() if res and str(res).strip() else NO_ANSWER

    if not text or not isinstance(text, str):
        return NO_ANSWER

    markers = ["\\boxed{", "\\fbox{"]
    last_start = -1
    marker_len = 0
    for m in markers:
        pos = text.rfind(m)
        if pos > last_start:
            last_start = pos
            marker_len = len(m)

    if last_start == -1:
        return NO_ANSWER

    idx = last_start + marker_len
    depth = 1
    chars: List[str] = []
    while idx < len(text) and depth > 0:
        ch = text[idx]
        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                chars.append(ch)
        else:
            chars.append(ch)
        idx += 1

    if depth != 0:
        return NO_ANSWER

    extracted = "".join(chars).strip()
    return extracted if extracted else NO_ANSWER


def _clean_Missing_ans(val: Any) -> str:
    if val is None:
        return NO_ANSWER
    if isinstance(val, float) and np.isnan(val):
        return NO_ANSWER
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null", "<na>", "no_answer", "__no_answer__"):
        return NO_ANSWER
    return s


def group_and_grade_math_question(
    raw_answers: Sequence[Any],
    gold_answer: Any,
) -> Tuple[List[str], np.ndarray]:
    """
    Forms answer equivalence groups for a MATH question using math-verify WITHOUT
    consulting the gold answer, then grades each group against the gold answer
    using the same equivalence rule.
    """
    if hasattr(answer_utils, "group_and_grade_math"):
        return answer_utils.group_and_grade_math(raw_answers, gold_answer)

    cleaned = [_clean_Missing_ans(a) for a in raw_answers]
    gold_str = _clean_Missing_ans(gold_answer)

    try:
        from math_verify import parse, verify
        has_mv = True
    except ImportError:
        has_mv = False

    def _norm_str(s: str) -> str:
        return re.sub(r"\s+", "", s.strip())

    if not has_mv:
        groups = [NO_ANSWER if c == NO_ANSWER else _norm_str(c) for c in cleaned]
        g_norm = _norm_str(gold_str)
        is_corr = np.array(
            [(g != NO_ANSWER) and (g == g_norm) for g in groups], dtype=bool
        )
        return groups, is_corr

    # Deduplicate raw strings within the question to minimize math_verify calls
    unique_strs: List[str] = []
    for c in cleaned:
        if c != NO_ANSWER and c not in unique_strs:
            unique_strs.append(c)

    parsed_cache: Dict[str, Any] = {}
    for u in unique_strs:
        try:
            parsed_cache[u] = parse(f"${u}$" if not u.startswith("$") else u)
        except Exception:
            parsed_cache[u] = None

    def _equiv(s1: str, s2: str) -> bool:
        if _norm_str(s1) == _norm_str(s2):
            return True
        p1, p2 = parsed_cache.get(s1), parsed_cache.get(s2)
        if not p1 or not p2:
            return False
        try:
            return bool(verify(p1, p2))
        except Exception:
            return False

    clusters: List[str] = []  # canonical string per cluster
    str_to_group: Dict[str, str] = {NO_ANSWER: NO_ANSWER}

    for u in unique_strs:
        matched_canon = None
        for canon in clusters:
            if _equiv(u, canon):
                matched_canon = canon
                break
        if matched_canon is None:
            clusters.append(u)
            str_to_group[u] = u
        else:
            str_to_group[u] = matched_canon

    # Grade each canonical cluster representative against gold_str
    try:
        gold_parsed = parse(f"${gold_str}$" if not gold_str.startswith("$") else gold_str)
    except Exception:
        gold_parsed = None

    canon_correct: Dict[str, bool] = {NO_ANSWER: False}
    for canon in clusters:
        if gold_str == NO_ANSWER:
            canon_correct[canon] = False
        elif _norm_str(canon) == _norm_str(gold_str):
            canon_correct[canon] = True
        elif gold_parsed and parsed_cache.get(canon):
            try:
                canon_correct[canon] = bool(verify(gold_parsed, parsed_cache[canon]))
            except Exception:
                canon_correct[canon] = False
        else:
            canon_correct[canon] = False

    ans_groups = [str_to_group.get(c, NO_ANSWER) for c in cleaned]
    is_corr = np.array([canon_correct.get(g, False) for g in ans_groups], dtype=bool)
    return ans_groups, is_corr


def normalize_mathvista_single(
    raw_ans: Any,
    q_meta: Dict[str, Any],
) -> str:
    """
    Normalizes a MathVista extracted or gold answer according to question_type,
    answer_type, choices, and precision rules from testmini.json WITHOUT consulting gold.
    Multiple-choice answers are represented as uppercase option letters (A, B, C, ...).
    """
    s = _clean_Missing_ans(raw_ans)
    if s == NO_ANSWER:
        return NO_ANSWER

    q_type = str(q_meta.get("question_type", "")).lower()
    ans_type = str(q_meta.get("answer_type", "")).lower()
    choices = q_meta.get("choices") or []
    precision = q_meta.get("precision", None)

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    if q_type == "multi_choice" or len(choices) > 0:
        s_clean = s.strip()
        # Direct option letter match: "A", "(B)", "Option C", "C."
        m = re.match(r"^(?:option\s*)?[\(\[\{]?([A-Za-z])[\)\]\}\.\:]?$", s_clean, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            idx = letters.find(letter)
            if 0 <= idx < max(len(choices), 26):
                return letter

        # Match against choice text strings
        s_low = s_clean.lower()
        for idx, ch_val in enumerate(choices):
            ch_str = str(ch_val).strip().lower()
            if s_low == ch_str:
                return letters[idx]

        # Numeric equivalence against choice values
        try:
            s_float = float(s_clean.replace(",", ""))
            for idx, ch_val in enumerate(choices):
                try:
                    if abs(s_float - float(str(ch_val).replace(",", ""))) < 1e-6:
                        return letters[idx]
                except Exception:
                    pass
        except Exception:
            pass

        # Leading letter fallback e.g. "A) 42"
        m2 = re.match(r"^([A-Za-z])[\)\.\:]\s+", s_clean)
        if m2:
            return m2.group(1).upper()
        return s_clean.upper() if len(s_clean) == 1 and s_clean.isalpha() else s_clean.lower()

    # Free-form integer / float / text
    if ans_type in ("integer", "float"):
        num_match = re.findall(r"[-+]?\d+(?:\.\d+)?", s.replace(",", ""))
        if num_match:
            try:
                val = float(num_match[-1])
                if ans_type == "integer":
                    return str(int(round(val)))
                if precision is not None:
                    prec = int(precision)
                    return f"{round(val, prec):.{prec}f}"
                return f"{round(val, 4):.4f}".rstrip("0").rstrip(".") or "0"
            except Exception:
                pass

    return re.sub(r"\s+", " ", s.strip().lower())


def group_and_grade_mathvista_question(
    raw_answers: Sequence[Any],
    gold_answer: Any,
    q_meta: Dict[str, Any],
) -> Tuple[List[str], np.ndarray]:
    """
    Normalizes MathVista candidate answers into answer groups without consulting gold,
    and grades against normalized gold using the same rules.
    """
    if hasattr(answer_utils, "group_and_grade_mathvista"):
        return answer_utils.group_and_grade_mathvista(raw_answers, gold_answer, q_meta)

    ans_groups = [normalize_mathvista_single(a, q_meta) for a in raw_answers]
    gold_norm = normalize_mathvista_single(gold_answer, q_meta)
    is_corr = np.array(
        [(g != NO_ANSWER) and (gold_norm != NO_ANSWER) and (g == gold_norm) for g in ans_groups],
        dtype=bool,
    )
    return ans_groups, is_corr


def _find_parquet_files(data_root: Path, is_vlm: bool) -> Dict[str, Path]:
    """Locates generations, probability_metrics, and attention_metrics parquet files."""
    gen_candidates = sorted(data_root.rglob("generations.parquet"))
    if not gen_candidates:
        raise FileNotFoundError(
            f"No generations.parquet found under {data_root}. "
            "Make sure ATML-PA3-Data is downloaded to --data-root."
        )

    selected_dir: Optional[Path] = None
    for cand in gen_candidates:
        parent_str = str(cand.parent).lower()
        has_attn = (cand.parent / "attention_metrics.parquet").exists()
        if is_vlm and ("vl" in parent_str or "mathvista" in parent_str or has_attn):
            selected_dir = cand.parent
            break
        if (not is_vlm) and ("vl" not in parent_str and "mathvista" not in parent_str and not has_attn):
            selected_dir = cand.parent
            break

    if selected_dir is None:
        selected_dir = gen_candidates[0].parent

    files = {
        "generations": selected_dir / "generations.parquet",
        "probability": selected_dir / "probability_metrics.parquet",
    }
    if is_vlm:
        files["attention"] = selected_dir / "attention_metrics.parquet"
    return files


def _pick_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    if required:
        raise KeyError(f"Could not find any of {candidates} in columns {list(df.columns)}")
    return None


def load_or_build_dataset_records(
    cfg: Task3Config,
    dataset_key: str,
    mode: str = "offline",
    data_dir: Optional[Path] = None,
    max_questions: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Loads and joins the supplied ATML-PA3-Data Parquet tables for `dataset_key`
    ('math' / 'math500' or 'mathvista'), verifies token-list alignment, applies
    seed 6304 lexicographical split permutation (200 dev, 800 eval), and caches
    CSR-packed rollout records.
    """
    is_vlm = dataset_key.lower() == "mathvista"
    cache_dir = cfg.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_sub{max_questions}" if max_questions is not None else ""
    pkl_path = cache_dir / f"{dataset_key}{suffix}_records.pkl"

    if pkl_path.exists() and mode != "rebuild":
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    root = Path(data_dir) if data_dir is not None else Path(getattr(cfg, "data_dir", "/kaggle/working/data/task3"))
    pq_files = _find_parquet_files(root, is_vlm=is_vlm)

    df_gen = pd.read_parquet(pq_files["generations"])
    df_prob = pd.read_parquet(pq_files["probability"])

    join_keys = [k for k in ("dataset", "sample_id", "rollout_id") if k in df_gen.columns and k in df_prob.columns]
    if not join_keys:
        join_keys = ["sample_id", "rollout_id"]

    df = pd.merge(df_gen, df_prob, on=join_keys, suffixes=("", "_prob"))
    if is_vlm:
        df_attn = pd.read_parquet(pq_files["attention"])
        attn_keys = [k for k in ("dataset", "sample_id", "rollout_id") if k in df.columns and k in df_attn.columns]
        df = pd.merge(df, df_attn, on=attn_keys, suffixes=("", "_attn"))

    # Load MathVista testmini.json metadata if VLM
    mv_meta: Dict[str, Any] = {}
    if is_vlm:
        meta_candidates = list(root.rglob("testmini.json")) + [
            Path("/kaggle/working/data/task3/testmini.json"),
            Path("/kaggle/working/testmini.json"),
            Path("testmini.json"),
        ]
        for mp in meta_candidates:
            if mp.exists():
                with open(mp, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                if isinstance(raw_meta, dict):
                    mv_meta = {str(k): v for k, v in raw_meta.items()}
                elif isinstance(raw_meta, list):
                    mv_meta = {str(item.get("pid", item.get("sample_id", i))): item for i, item in enumerate(raw_meta)}
                break

    qid_col = _pick_col(df, ["sample_id", "question_id", "qid", "pid", "id"])
    rid_col = _pick_col(df, ["rollout_id", "candidate_id", "rollout_idx"])
    ext_col = _pick_col(df, ["extracted_answer", "extracted_answers", "pred_answer", "extracted"])
    gold_col = _pick_col(df, ["gold_answer", "gold_answers", "gold", "target", "ground_truth", "answer"])
    len_col = _pick_col(df, ["length", "lengths", "num_tokens", "token_count", "gen_length"], required=False)
    logp_col = _pick_col(df, ["logp", "sampled_token_logp", "token_logp", "logprobs"])
    kl_col = _pick_col(df, ["kl_u_p", "kl", "self_certainty"])
    prompt_col = _pick_col(df, ["problem", "question", "prompt", "query", "input_text", "prompt_text"], required=False)
    text_col = _pick_col(df, ["text", "response", "generation", "output_text", "completion"], required=False)

    if is_vlm:
        img_attn_col = _pick_col(df, ["image", "image_attention", "attn_image", "a_image", "vision"])
        prm_attn_col = _pick_col(df, ["prompt_text", "prompt", "prompt_attention", "attn_prompt", "a_prompt"])
        gen_attn_col = _pick_col(df, ["generated_text", "generated", "gen_text", "generated_attention", "attn_generated", "a_generated"])

    # Lexicographical sort of question IDs, then permute with default_rng(6304)
    df["_qid_str"] = df[qid_col].astype(str)
    all_qids_sorted = sorted(df["_qid_str"].unique().tolist())
    split_rng = np.random.default_rng(cfg.seed)
    permuted_qids = list(all_qids_sorted)
    split_rng.shuffle(permuted_qids)

    dev_qids_list = permuted_qids[:200]
    eval_qids_list = permuted_qids[200:]
    live_math_50_qids = eval_qids_list[:50] if not is_vlm else []

    if max_questions is not None:
        half_q = max(max_questions // 2, 2)
        active_dev = set(dev_qids_list[:half_q])
        active_eval = set(eval_qids_list[:half_q])
        keep_qids = active_dev | active_eval
        df = df[df["_qid_str"].isin(keep_qids)].copy()
    else:
        active_dev = set(dev_qids_list)

    # Save split IDs and scorer metadata
    split_dir = cfg.output_dir / "step1"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_info = {
        "dataset": dataset_key,
        "seed": cfg.seed,
        "math_verify_version": _get_math_verify_version(),
        "total_questions": len(permuted_qids),
        "dev_question_ids": dev_qids_list,
        "eval_question_ids": eval_qids_list,
        "live_step4_50_question_ids": live_math_50_qids,
    }
    with open(split_dir / f"{dataset_key}_split_ids.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    records: Dict[str, Dict[str, Any]] = {}
    order_rng = np.random.default_rng(cfg.seed)

    total_rollouts_checked = 0
    alignment_mismatches = 0

    grouped = df.groupby("_qid_str", sort=False)
    for qid_str, group in tqdm(grouped, desc=f"Building {dataset_key} records"):
        split_label = "dev" if qid_str in active_dev else "eval"
        group_sorted = group.sort_values(rid_col).head(cfg.num_rollouts)
        num_r = len(group_sorted)

        raw_exts = group_sorted[ext_col].tolist()
        gold_val = group_sorted[gold_col].iloc[0]

        q_meta = mv_meta.get(qid_str, {}) if is_vlm else {}
        if is_vlm:
            # Merge any metadata columns present in the parquet row itself
            row0 = group_sorted.iloc[0]
            for mk in ("question_type", "answer_type", "choices", "precision", "unit"):
                if mk not in q_meta and mk in row0 and row0[mk] is not None:
                    val = row0[mk]
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    q_meta[mk] = val
            ans_groups, is_corr = group_and_grade_mathvista_question(raw_exts, gold_val, q_meta)
        else:
            ans_groups, is_corr = group_and_grade_math_question(raw_exts, gold_val)

        lengths = np.zeros(num_r, dtype=np.int32)
        L_arr = np.zeros(num_r, dtype=np.float64)
        K_arr = np.zeros(num_r, dtype=np.float64)
        V_arr = np.zeros(num_r, dtype=np.float64)
        KV_arr = np.zeros(num_r, dtype=np.float64)
        cov_kv_arr = np.zeros(num_r, dtype=np.float64)
        corr_kv_arr = np.zeros(num_r, dtype=np.float64)
        k_tokens_list: List[np.ndarray] = []
        v_tokens_list: List[np.ndarray] = []

        for r_idx, (_, row) in enumerate(group_sorted.iterrows()):
            total_rollouts_checked += 1
            logp = np.asarray(row[logp_col], dtype=np.float64)
            kl = np.asarray(row[kl_col], dtype=np.float64)

            T_tok = len(logp)
            if len_col is not None and pd.notna(row[len_col]):
                reported_len = int(row[len_col])
            else:
                reported_len = T_tok

            if len(logp) != len(kl) or (reported_len > 0 and len(logp) != reported_len):
                alignment_mismatches += 1

            T_eff = max(min(len(logp), len(kl)), 1)
            logp = logp[:T_eff]
            kl = kl[:T_eff]

            lengths[r_idx] = reported_len if reported_len > 0 else T_eff
            L_arr[r_idx] = float(np.mean(logp)) if len(logp) > 0 else 0.0
            K_arr[r_idx] = float(np.mean(kl)) if len(kl) > 0 else 0.0

            if is_vlm:
                a_img = np.asarray(row[img_attn_col], dtype=np.float64)[:T_eff]
                a_prm = np.asarray(row[prm_attn_col], dtype=np.float64)[:T_eff]
                a_gen = np.asarray(row[gen_attn_col], dtype=np.float64)[:T_eff]
                if len(a_img) != len(kl):
                    alignment_mismatches += 1
                denom = a_img + a_prm + a_gen
                v_t = np.divide(
                    a_img,
                    denom,
                    out=np.zeros_like(a_img, dtype=np.float64),
                    where=denom > 1e-12,
                )
                V_arr[r_idx] = float(np.mean(v_t)) if len(v_t) > 0 else 0.0
                if len(kl) > 0 and len(v_t) == len(kl):
                    KV_arr[r_idx] = float(np.mean(kl * v_t))
                    cov_kv = KV_arr[r_idx] - K_arr[r_idx] * V_arr[r_idx]
                    cov_kv_arr[r_idx] = float(cov_kv)
                    std_k = float(np.std(kl))
                    std_v = float(np.std(v_t))
                    if std_k > 1e-12 and std_v > 1e-12:
                        corr_kv_arr[r_idx] = float(np.clip(cov_kv / (std_k * std_v), -1.0, 1.0))
                v_tokens_list.append(v_t)
            else:
                v_tokens_list.append(np.zeros_like(kl, dtype=np.float64))

            k_tokens_list.append(kl)

        tok_lens = np.array([len(a) for a in k_tokens_list], dtype=np.int64)
        offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(tok_lens)])
        kl_flat = np.concatenate(k_tokens_list).astype(np.float64) if offsets[-1] > 0 else np.zeros(0, dtype=np.float64)
        v_flat = np.concatenate(v_tokens_list).astype(np.float64) if offsets[-1] > 0 else np.zeros(0, dtype=np.float64)

        orders = np.stack(
            [order_rng.permutation(num_r).astype(np.int32) for _ in range(cfg.num_orders)],
            axis=0,
        )

        row_first = group_sorted.iloc[0]
        problem_text = ""
        if prompt_col is not None and not is_vlm and pd.notna(row_first[prompt_col]):
            problem_text = str(row_first[prompt_col])
        elif text_col is not None and pd.notna(row_first[text_col]):
            problem_text = str(row_first[text_col])

        records[qid_str] = {
            "qid": qid_str,
            "dataset": dataset_key,
            "split": split_label,
            "gold_answer": str(gold_val),
            "problem_text": problem_text,
            "raw_extracted_answers": [_clean_Missing_ans(x) for x in raw_exts],
            "answer_groups": ans_groups,
            "is_correct": is_corr,
            "lengths": lengths,
            "L": L_arr,
            "K": K_arr,
            "V": V_arr,
            "KV": KV_arr,
            "cov_kv": cov_kv_arr,
            "corr_kv": corr_kv_arr,
            "offsets": offsets,
            "kl_flat": kl_flat,
            "v_flat": v_flat,
            "orders": orders,
        }

    alignment_report = {
        "dataset": dataset_key,
        "total_questions": len(records),
        "total_rollouts_checked": total_rollouts_checked,
        "alignment_mismatches": alignment_mismatches,
        "math_verify_version": _get_math_verify_version(),
    }
    with open(split_dir / f"{dataset_key}_alignment_verification.json", "w", encoding="utf-8") as f:
        json.dump(alignment_report, f, indent=2)

    with open(pkl_path, "wb") as f:
        pickle.dump(records, f)

    return records


def _resolve_math_problem_texts(
    data_root: Path,
    target_qids: Sequence[str],
    records: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """
    Retrieves the original problem statement for each of the 50 MATH evaluation questions
    from the supplied Parquet / JSON / README files.
    """
    pq_files = _find_parquet_files(data_root, is_vlm=False)
    df_gen = pd.read_parquet(pq_files["generations"])
    qid_col = _pick_col(df_gen, ["sample_id", "question_id", "qid", "id"])
    df_gen["_qid_str"] = df_gen[qid_col].astype(str)
    sub_df = df_gen[df_gen["_qid_str"].isin(set(target_qids))].drop_duplicates("_qid_str")

    prob_col = _pick_col(
        sub_df,
        ["problem", "question", "prompt", "query", "input", "problem_text", "question_text"],
        required=False,
    )
    problems: Dict[str, str] = {}
    if prob_col is not None:
        for _, row in sub_df.iterrows():
            problems[str(row["_qid_str"])] = str(row[prob_col])

    # Check if a separate questions/problems JSON or parquet exists in data_root
    if len(problems) < len(target_qids):
        for p_file in data_root.rglob("*.json*"):
            if "testmini" in p_file.name.lower():
                continue
            try:
                df_q = pd.read_json(p_file, lines=p_file.suffix == ".jsonl")
                q_c = _pick_col(df_q, ["sample_id", "question_id", "id", "qid"], required=False)
                p_c = _pick_col(df_q, ["problem", "question", "prompt"], required=False)
                if q_c and p_c:
                    for _, r in df_q.iterrows():
                        k = str(r[q_c])
                        if k in target_qids and k not in problems:
                            problems[k] = str(r[p_c])
            except Exception:
                pass

    for qid in target_qids:
        if qid not in problems or not problems[qid].strip():
            problems[qid] = records[qid].get("problem_text", f"Solve mathematical problem ID {qid}.")
    return problems


def run_step4_live_math_experiment(
    cfg: Task3Config,
    math_records: Dict[str, Dict[str, Any]],
    data_dir: Optional[Path] = None,
    num_questions: int = 50,
    smoke_mode: bool = False,
) -> Dict[str, Any]:
    """
    Step 4: Live Generation on the first 50 MATH evaluation questions (in the seed 6304
    permutation) comparing three strategies under a 3,072-token total output allowance:
      1. `single_long`: 1 response capped at 3,072 new tokens.
      2. `parallel_3`: 3 independent responses capped at 1,024 new tokens each (self-consistency).
      3. `sequential_refinement`: reuses the first 1,024-token parallel response as an
         assistant turn, appends the review prompt, and generates 1 critique-and-revision
         response capped at 2,048 new tokens.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    step4_dir = cfg.output_dir / "step4"
    step4_dir.mkdir(parents=True, exist_ok=True)

    root = Path(data_dir) if data_dir is not None else Path(getattr(cfg, "data_dir", "/kaggle/working/data/task3"))
    split_json = cfg.output_dir / "step1" / f"{next(iter(math_records.values()))['dataset']}_split_ids.json"
    if split_json.exists():
        with open(split_json, "r", encoding="utf-8") as f:
            s_info = json.load(f)
        eval_ordered = [q for q in s_info.get("live_step4_50_question_ids", []) if q in math_records]
    else:
        eval_ordered = [qid for qid, r in math_records.items() if r["split"] == "eval"]

    target_qids = eval_ordered[: (1 if smoke_mode else min(num_questions, len(eval_ordered)))]
    problem_map = _resolve_math_problem_texts(root, target_qids, math_records)

    model_name = getattr(cfg, "live_model_name", "Qwen/Qwen2.5-3B-Instruct")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[Step 4 Live] Loading {model_name} in {dtype} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    cap_long = 64 if smoke_mode else 3072
    cap_short = 32 if smoke_mode else 1024
    cap_refine = 48 if smoke_mode else 2048

    sys_prompt = "You are a helpful assistant."
    user_suffix = "\nThink step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
    refine_instruction = (
        "Review the solution above. Identify and correct any errors; "
        "if it is correct, retain its answer. Give your final answer inside \\boxed{}."
    )

    def _generate_once(messages: List[Dict[str, str]], max_new_toks: int, call_seed: int) -> Dict[str, Any]:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        in_toks = int(inputs["input_ids"].shape[1])

        torch.manual_seed(call_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(call_seed)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.0,
                max_new_tokens=max_new_toks,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = float(time.perf_counter() - t0)
        peak_mem_mb = (
            float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
            if torch.cuda.is_available()
            else 0.0
        )

        gen_ids = out[0, in_toks:]
        out_toks = int(gen_ids.shape[0])
        hit_eos = bool(len(gen_ids) > 0 and int(gen_ids[-1].item()) == int(tokenizer.eos_token_id))
        truncated = bool(out_toks >= max_new_toks and not hit_eos)
        resp_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        extracted = extract_last_boxed_balanced(resp_text)

        return {
            "prompt": prompt_text,
            "response": resp_text,
            "extracted_answer": extracted,
            "has_extracted": bool(extracted != NO_ANSWER),
            "truncated": truncated,
            "input_tokens": in_toks,
            "output_tokens": out_toks,
            "runtime_sec": elapsed,
            "peak_gpu_mb": peak_mem_mb,
            "seed": call_seed,
        }

    per_question_logs: List[Dict[str, Any]] = []

    for q_idx, qid in enumerate(tqdm(target_qids, desc="Step 4 Live MATH Generation")):
        gold_ans = math_records[qid]["gold_answer"]
        prob_text = problem_map[qid]
        base_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prob_text + user_suffix},
        ]

        base_seed = int(cfg.seed) + q_idx * 100

        # 1. Single longer response (cap 3,072)
        res_long = _generate_once(base_messages, cap_long, base_seed + 1)

        # 2. Parallel sampling: 3 independent responses (cap 1,024 each)
        res_p1 = _generate_once(base_messages, cap_short, base_seed + 2)
        res_p2 = _generate_once(base_messages, cap_short, base_seed + 3)
        res_p3 = _generate_once(base_messages, cap_short, base_seed + 4)

        # 3. Sequential refinement: reuse res_p1 as assistant message + critique/revision (cap 2,048)
        refine_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prob_text + user_suffix},
            {"role": "assistant", "content": res_p1["response"]},
            {"role": "user", "content": refine_instruction},
        ]
        res_ref = _generate_once(refine_messages, cap_refine, base_seed + 5)

        # Equivalence grouping & grading across the 5 generated answers for this question
        all_ext = [
            res_long["extracted_answer"],
            res_p1["extracted_answer"],
            res_p2["extracted_answer"],
            res_p3["extracted_answer"],
            res_ref["extracted_answer"],
        ]
        all_groups, all_corr = group_and_grade_math_question(all_ext, gold_ans)

        long_group, long_corr = all_groups[0], bool(all_corr[0])
        p_groups = all_groups[1:4]
        p_corrs = all_corr[1:4]
        init_group, init_corr = all_groups[1], bool(all_corr[1])
        ref_group, ref_corr = all_groups[4], bool(all_corr[4])

        # Self-consistency vote over the 3 parallel candidates (earliest tie-break)
        _, par_sc_corr, _, _ = select_weighted_consensus(
            np.array([0, 1, 2], dtype=np.int32),
            np.ones(3, dtype=np.float64),
            p_groups,
            p_corrs,
        )

        # Transition from shared initial response (p1) to revised response
        if (not init_corr) and ref_corr:
            transition = "wrong_to_correct"
        elif init_corr and (not ref_corr):
            transition = "correct_to_wrong"
        elif init_corr and ref_corr:
            transition = "correct_to_correct"
        else:
            transition = "wrong_to_wrong"

        per_question_logs.append(
            {
                "qid": qid,
                "gold_answer": gold_ans,
                "problem": prob_text,
                "single_long": {
                    **res_long,
                    "answer_group": long_group,
                    "is_correct": long_corr,
                },
                "parallel_3": {
                    "candidates": [res_p1, res_p2, res_p3],
                    "candidate_groups": p_groups,
                    "candidate_correct": [bool(x) for x in p_corrs],
                    "is_correct": bool(par_sc_corr),
                    "input_tokens": res_p1["input_tokens"] + res_p2["input_tokens"] + res_p3["input_tokens"],
                    "output_tokens": res_p1["output_tokens"] + res_p2["output_tokens"] + res_p3["output_tokens"],
                    "runtime_sec": res_p1["runtime_sec"] + res_p2["runtime_sec"] + res_p3["runtime_sec"],
                    "peak_gpu_mb": max(res_p1["peak_gpu_mb"], res_p2["peak_gpu_mb"], res_p3["peak_gpu_mb"]),
                    "any_truncated": bool(res_p1["truncated"] or res_p2["truncated"] or res_p3["truncated"]),
                    "truncation_rate": float(np.mean([res_p1["truncated"], res_p2["truncated"], res_p3["truncated"]])),
                    "extraction_rate": float(np.mean([res_p1["has_extracted"], res_p2["has_extracted"], res_p3["has_extracted"]])),
                },
                "sequential_refinement": {
                    "initial": res_p1,
                    "revision": res_ref,
                    "initial_group": init_group,
                    "initial_correct": init_corr,
                    "revised_group": ref_group,
                    "is_correct": ref_corr,
                    "transition": transition,
                    # Charge the shared initial response + revision prompt & critique/revision tokens
                    "input_tokens": res_p1["input_tokens"] + res_ref["input_tokens"],
                    "output_tokens": res_p1["output_tokens"] + res_ref["output_tokens"],
                    "runtime_sec": res_p1["runtime_sec"] + res_ref["runtime_sec"],
                    "peak_gpu_mb": max(res_p1["peak_gpu_mb"], res_ref["peak_gpu_mb"]),
                    "initial_truncated": res_p1["truncated"],
                    "revision_truncated": res_ref["truncated"],
                    "revision_has_extracted": res_ref["has_extracted"],
                },
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate summary metrics across the 50 questions
    n_q = max(len(per_question_logs), 1)
    summary_rows = [
        {
            "strategy": "initial_1024_shared",
            "num_questions": len(per_question_logs),
            "accuracy": float(np.mean([q["sequential_refinement"]["initial_correct"] for q in per_question_logs])),
            "mean_input_tokens": float(np.mean([q["sequential_refinement"]["initial"]["input_tokens"] for q in per_question_logs])),
            "mean_output_tokens": float(np.mean([q["sequential_refinement"]["initial"]["output_tokens"] for q in per_question_logs])),
            "mean_total_tokens": float(
                np.mean(
                    [
                        q["sequential_refinement"]["initial"]["input_tokens"]
                        + q["sequential_refinement"]["initial"]["output_tokens"]
                        for q in per_question_logs
                    ]
                )
            ),
            "mean_runtime_sec": float(np.mean([q["sequential_refinement"]["initial"]["runtime_sec"] for q in per_question_logs])),
            "peak_gpu_mb": float(np.max([q["sequential_refinement"]["initial"]["peak_gpu_mb"] for q in per_question_logs])),
            "extraction_rate": float(np.mean([q["sequential_refinement"]["initial"]["has_extracted"] for q in per_question_logs])),
            "truncation_rate": float(np.mean([q["sequential_refinement"]["initial"]["truncated"] for q in per_question_logs])),
        },
        {
            "strategy": "single_long_3072",
            "num_questions": len(per_question_logs),
            "accuracy": float(np.mean([q["single_long"]["is_correct"] for q in per_question_logs])),
            "mean_input_tokens": float(np.mean([q["single_long"]["input_tokens"] for q in per_question_logs])),
            "mean_output_tokens": float(np.mean([q["single_long"]["output_tokens"] for q in per_question_logs])),
            "mean_total_tokens": float(
                np.mean([q["single_long"]["input_tokens"] + q["single_long"]["output_tokens"] for q in per_question_logs])
            ),
            "mean_runtime_sec": float(np.mean([q["single_long"]["runtime_sec"] for q in per_question_logs])),
            "peak_gpu_mb": float(np.max([q["single_long"]["peak_gpu_mb"] for q in per_question_logs])),
            "extraction_rate": float(np.mean([q["single_long"]["has_extracted"] for q in per_question_logs])),
            "truncation_rate": float(np.mean([q["single_long"]["truncated"] for q in per_question_logs])),
        },
        {
            "strategy": "parallel_3x1024",
            "num_questions": len(per_question_logs),
            "accuracy": float(np.mean([q["parallel_3"]["is_correct"] for q in per_question_logs])),
            "mean_input_tokens": float(np.mean([q["parallel_3"]["input_tokens"] for q in per_question_logs])),
            "mean_output_tokens": float(np.mean([q["parallel_3"]["output_tokens"] for q in per_question_logs])),
            "mean_total_tokens": float(
                np.mean([q["parallel_3"]["input_tokens"] + q["parallel_3"]["output_tokens"] for q in per_question_logs])
            ),
            "mean_runtime_sec": float(np.mean([q["parallel_3"]["runtime_sec"] for q in per_question_logs])),
            "peak_gpu_mb": float(np.max([q["parallel_3"]["peak_gpu_mb"] for q in per_question_logs])),
            "extraction_rate": float(np.mean([q["parallel_3"]["extraction_rate"] for q in per_question_logs])),
            "truncation_rate": float(np.mean([q["parallel_3"]["truncation_rate"] for q in per_question_logs])),
        },
        {
            "strategy": "sequential_refinement_1024_2048",
            "num_questions": len(per_question_logs),
            "accuracy": float(np.mean([q["sequential_refinement"]["is_correct"] for q in per_question_logs])),
            "mean_input_tokens": float(np.mean([q["sequential_refinement"]["input_tokens"] for q in per_question_logs])),
            "mean_output_tokens": float(np.mean([q["sequential_refinement"]["output_tokens"] for q in per_question_logs])),
            "mean_total_tokens": float(
                np.mean(
                    [
                        q["sequential_refinement"]["input_tokens"]
                        + q["sequential_refinement"]["output_tokens"]
                        for q in per_question_logs
                    ]
                )
            ),
            "mean_runtime_sec": float(np.mean([q["sequential_refinement"]["runtime_sec"] for q in per_question_logs])),
            "peak_gpu_mb": float(np.max([q["sequential_refinement"]["peak_gpu_mb"] for q in per_question_logs])),
            "extraction_rate": float(np.mean([q["sequential_refinement"]["revision_has_extracted"] for q in per_question_logs])),
            "truncation_rate": float(np.mean([q["sequential_refinement"]["revision_truncated"] for q in per_question_logs])),
        },
    ]

    transitions = [q["sequential_refinement"]["transition"] for q in per_question_logs]
    transition_summary = {
        "wrong_to_correct_count": int( sum(t == "wrong_to_correct" for t in transitions) ),
        "correct_to_wrong_count": int( sum(t == "correct_to_wrong" for t in transitions) ),
        "correct_to_correct_count": int( sum(t == "correct_to_correct" for t in transitions) ),
        "wrong_to_wrong_count": int( sum(t == "wrong_to_wrong" for t in transitions) ),
        "wrong_to_correct_rate": float( sum(t == "wrong_to_correct" for t in transitions) / n_q ),
        "correct_to_wrong_rate": float( sum(t == "correct_to_wrong" for t in transitions) / n_q ),
        "net_accuracy_gain": float(
            (sum(t == "wrong_to_correct" for t in transitions) - sum(t == "correct_to_wrong" for t in transitions)) / n_q
        ),
    }

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(step4_dir / "math_step4_live_summary.csv", index=False)

    payload = {
        "model": model_name,
        "num_questions": len(per_question_logs),
        "strategy_summary": summary_rows,
        "refinement_transitions": transition_summary,
    }
    with open(step4_dir / "math_step4_live_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(step4_dir / "math_step4_live_generations.jsonl", "w", encoding="utf-8") as f:
        for item in per_question_logs:
            f.write(json.dumps(item) + "\n")

    return payload


def extract_step6_qualitative_cases(
    cfg: Task3Config,
    all_records: Dict[str, Dict[str, Dict[str, Any]]],
    step4_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Step 6: Extracts representative cases of (a) selection disagreement where RWC/VisConf
    recovers a correct answer missed by majority voting, (b) sequential refinement
    transitions from Step 4, and (c) premature adaptive stopping vs pool correctness rate.
    """
    step6_dir = cfg.output_dir / "step6"
    step6_dir.mkdir(parents=True, exist_ok=True)
    full_prefix = np.arange(cfg.num_rollouts, dtype=np.int32)

    cases: Dict[str, Any] = {"selection_disagreements": {}, "premature_stopping": {}}

    for ds_key, records in all_records.items():
        is_vlm = ds_key.lower() == "mathvista"
        eval_qids = [qid for qid, r in records.items() if r["split"] == "eval"]
        disagree_list = []

        for qid in eval_qids:
            rec = records[qid]
            ans_groups = rec["answer_groups"]
            is_corr = rec["is_correct"]
            sc_g, sc_ok, _, _ = select_weighted_consensus(
                full_prefix, np.ones(cfg.num_rollouts, dtype=np.float64), ans_groups, is_corr
            )
            if is_vlm:
                _, scores, _, _ = compute_prefix_visconf(rec, full_prefix)
            else:
                scores = rec["K"]
            rwc_g, rwc_ok, _, _, rwc_margin = select_rwc(
                full_prefix, scores, ans_groups, is_corr
            )
            if (not sc_ok) and rwc_ok:
                disagree_list.append(
                    {
                        "qid": qid,
                        "gold_answer": rec["gold_answer"],
                        "sc_selected": sc_g,
                        "rwc_selected": rwc_g,
                        "rwc_margin": float(rwc_margin),
                        "pool_correct_fraction": float(np.mean(is_corr)),
                    }
                )
            if len(disagree_list) >= 5:
                break
        cases["selection_disagreements"][ds_key] = disagree_list

    out_file = step6_dir / "step6_qualitative_cases.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)
    return cases
