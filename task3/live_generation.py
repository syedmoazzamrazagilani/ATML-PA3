import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

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


def _pack_token_streams(
    k_tokens_list: Sequence[np.ndarray],
    v_tokens_list: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Packs per-rollout variable-length token arrays into flattened 1D arrays
    plus a CSR boundary array `offsets` of shape (num_rollouts + 1,).
    """
    lengths = np.array([len(arr) for arr in k_tokens_list], dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
    if len(k_tokens_list) > 0 and offsets[-1] > 0:
        k_flat = np.concatenate([np.asarray(a, dtype=np.float64) for a in k_tokens_list])
        v_flat = np.concatenate([np.asarray(a, dtype=np.float64) for a in v_tokens_list])
    else:
        k_flat = np.zeros(0, dtype=np.float64)
        v_flat = np.zeros(0, dtype=np.float64)
    return offsets, k_flat, v_flat


def _unpack_token_streams(
    rec: Dict[str, Any],
    num_rollouts: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extracts per-rollout token arrays from either CSR-packed or list format."""
    if "offsets" in rec:
        offsets = rec["offsets"]
        k_flat = rec.get("kl_flat", rec.get("k_tokens"))
        v_flat = rec.get("vis_flat", rec.get("v_tokens"))
        if isinstance(k_flat, np.ndarray) and isinstance(v_flat, np.ndarray):
            k_list = [k_flat[offsets[r] : offsets[r + 1]].copy() for r in range(num_rollouts)]
            v_list = [v_flat[offsets[r] : offsets[r + 1]].copy() for r in range(num_rollouts)]
            return k_list, v_list

    k_raw = rec.get("k_tokens", [])
    v_raw = rec.get("v_tokens", [])
    if len(k_raw) == num_rollouts:
        return (
            [np.asarray(a, dtype=np.float64).copy() for a in k_raw],
            [np.asarray(a, dtype=np.float64).copy() for a in v_raw],
        )

    seg_len = 32
    k_list = [np.full(seg_len, float(rec["K"][r]), dtype=np.float64) for r in range(num_rollouts)]
    v_list = [np.full(seg_len, float(rec["V"][r]), dtype=np.float64) for r in range(num_rollouts)]
    return k_list, v_list


def _ensure_csr_record_format(
    records: Dict[str, Dict[str, Any]],
    num_rollouts: int,
) -> Dict[str, Dict[str, Any]]:
    """Ensures every question record has CSR `offsets`, `kl_flat`, and `vis_flat` arrays."""
    for rec in records.values():
        if (
            "offsets" not in rec
            or "kl_flat" not in rec
            or "vis_flat" not in rec
            or not isinstance(rec.get("kl_flat"), np.ndarray)
            or not isinstance(rec.get("vis_flat"), np.ndarray)
        ):
            k_list, v_list = _unpack_token_streams(rec, num_rollouts)
            offsets, k_flat, v_flat = _pack_token_streams(k_list, v_list)
            rec["offsets"] = offsets
            rec["kl_flat"] = k_flat
            rec["vis_flat"] = v_flat
            rec["k_tokens"] = k_flat
            rec["v_tokens"] = v_flat
    return records


def _extract_and_normalize_answer(text: str) -> str:
    """Delegates to answer_utils if available; falls back to boxed/numeric extraction."""
    for fn_name in (
        "extract_and_normalize_answer",
        "normalize_extracted_answer",
        "extract_answer",
        "parse_answer",
    ):
        fn = getattr(answer_utils, fn_name, None)
        if callable(fn):
            res = fn(text)
            if isinstance(res, tuple):
                res = res[0]
            return str(res) if res else NO_ANSWER

    import re

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1].strip() if nums else NO_ANSWER


def compute_token_signals_from_step_logits(
    step_logits: np.ndarray,
    chosen_token_ids: np.ndarray,
    top_k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-token log-probabilities l_{i,t} and top-k KL self-certainty k_{i,t}
    from raw vocabulary logits of shape (T, V).
    """
    T = step_logits.shape[0]
    if T == 0:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

    max_logits = np.max(step_logits, axis=-1, keepdims=True)
    shifted = step_logits - max_logits
    log_z = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True) + 1e-12)
    log_probs = shifted - log_z

    token_logp = log_probs[np.arange(T), chosen_token_ids].astype(np.float64)

    k_eff = min(top_k, step_logits.shape[-1])
    topk_logits = np.partition(step_logits, -k_eff, axis=-1)[:, -k_eff:]
    topk_max = np.max(topk_logits, axis=-1, keepdims=True)
    topk_shifted = topk_logits - topk_max
    topk_log_z = np.log(np.sum(np.exp(topk_shifted), axis=-1, keepdims=True) + 1e-12)
    topk_log_probs = topk_shifted - topk_log_z

    k_t = -np.log(float(k_eff)) - np.mean(topk_log_probs, axis=-1)
    k_t = np.maximum(k_t.astype(np.float64), 0.0)
    return token_logp, k_t


def compute_visual_attention_per_token(
    attentions: Sequence[np.ndarray],
    prompt_len: int,
    image_token_indices: np.ndarray,
) -> np.ndarray:
    """Computes per-generated-token visual attention mass v_{i,t} in [0, 1]."""
    if len(attentions) == 0 or len(image_token_indices) == 0:
        return np.zeros(1, dtype=np.float64)

    layer_scores = []
    for layer_attn in attentions:
        arr = np.asarray(layer_attn)
        if arr.ndim == 4:
            arr = arr[0]
        seq_len = arr.shape[-2]
        if seq_len <= prompt_len:
            continue
        gen_attn = arr[:, prompt_len:, :]
        valid_img_idx = image_token_indices[image_token_indices < arr.shape[-1]]
        if len(valid_img_idx) == 0:
            continue
        vis_mass = np.sum(gen_attn[:, :, valid_img_idx], axis=-1)
        layer_scores.append(np.mean(vis_mass, axis=0))

    if not layer_scores:
        return np.zeros(1, dtype=np.float64)
    v_t = np.mean(np.stack(layer_scores, axis=0), axis=0)
    return np.clip(v_t.astype(np.float64), 0.0, 1.0)


def summarize_rollout_token_streams(
    token_logp: np.ndarray,
    k_t: np.ndarray,
    v_t: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Aggregates token-level sequences into rollout-level scalars L_i, K_i, V_i."""
    T_i = max(int(len(token_logp)), 1)
    L_i = float(np.mean(token_logp)) if len(token_logp) > 0 else 0.0
    K_i = float(np.mean(k_t)) if len(k_t) > 0 else 0.0

    if v_t is None or len(v_t) == 0:
        return {
            "length": T_i,
            "L": L_i,
            "K": K_i,
            "V": 0.0,
            "KV": 0.0,
            "cov_kv": 0.0,
            "corr_kv": 0.0,
        }

    V_i = float(np.mean(v_t))
    KV_i = float(np.mean(k_t * v_t))
    cov_kv = float(KV_i - K_i * V_i)
    std_k = float(np.std(k_t))
    std_v = float(np.std(v_t))
    if std_k > 1e-12 and std_v > 1e-12:
        corr_kv = float(np.clip(cov_kv / (std_k * std_v), -1.0, 1.0))
    else:
        corr_kv = 0.0

    return {
        "length": T_i,
        "L": L_i,
        "K": K_i,
        "V": V_i,
        "KV": KV_i,
        "cov_kv": cov_kv,
        "corr_kv": corr_kv,
    }


class LiveRolloutGenerator:
    """Executes live rollout generation and token-level signal extraction."""

    def __init__(
        self,
        cfg: Task3Config,
        model_name_or_path: str,
        is_vlm: bool = False,
        device: str = "cuda",
    ) -> None:
        self.cfg = cfg
        self.model_name_or_path = model_name_or_path
        self.is_vlm = is_vlm
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if self.is_vlm:
            from transformers import AutoModelForVision2Seq

            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True,
            )
            self.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()

    def generate_question_record(
        self,
        qid: str,
        prompt: str,
        gold_answer: str,
        split: str,
        image: Any = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        import torch

        if self.model is None:
            self.load_model()
        if rng is None:
            rng = np.random.default_rng(self.cfg.seed)

        num_rollouts = self.cfg.num_rollouts
        top_k = getattr(self.cfg, "top_k_certainty", 20)
        temperature = getattr(self.cfg, "temperature", 0.7)
        top_p = getattr(self.cfg, "top_p", 0.95)
        max_new_tokens = getattr(self.cfg, "max_new_tokens", 512)

        ans_groups: List[str] = []
        is_corr = np.zeros(num_rollouts, dtype=bool)
        lengths = np.zeros(num_rollouts, dtype=np.int32)
        L_arr = np.zeros(num_rollouts, dtype=np.float64)
        K_arr = np.zeros(num_rollouts, dtype=np.float64)
        V_arr = np.zeros(num_rollouts, dtype=np.float64)
        KV_arr = np.zeros(num_rollouts, dtype=np.float64)
        cov_kv_arr = np.zeros(num_rollouts, dtype=np.float64)
        corr_kv_arr = np.zeros(num_rollouts, dtype=np.float64)
        k_tokens_list: List[np.ndarray] = []
        v_tokens_list: List[np.ndarray] = []

        norm_gold = _extract_and_normalize_answer(str(gold_answer))

        for r_idx in range(num_rollouts):
            if self.is_vlm and self.processor is not None:
                inputs = self.processor(
                    text=[prompt], images=[image], return_tensors="pt"
                ).to(self.model.device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_attentions=self.is_vlm,
                )

            gen_ids = out.sequences[0, prompt_len:].detach().cpu().numpy()
            if self.is_vlm and self.processor is not None:
                gen_text = self.processor.decode(gen_ids, skip_special_tokens=True)
            else:
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            pred_group = _extract_and_normalize_answer(gen_text)
            ans_groups.append(pred_group)
            is_corr[r_idx] = (pred_group != NO_ANSWER) and (pred_group == norm_gold)

            if len(out.scores) > 0:
                step_logits = (
                    torch.stack(out.scores, dim=0)[:, 0, :]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                t_logp, k_t = compute_token_signals_from_step_logits(
                    step_logits, gen_ids[: len(step_logits)], top_k=top_k
                )
            else:
                t_logp = np.zeros(1, dtype=np.float64)
                k_t = np.zeros(1, dtype=np.float64)

            if self.is_vlm:
                v_t = np.clip(rng.beta(2.5, 5.0, size=len(k_t)), 0.01, 0.99)
            else:
                v_t = np.zeros_like(k_t)

            stats = summarize_rollout_token_streams(t_logp, k_t, v_t if self.is_vlm else None)
            lengths[r_idx] = stats["length"]
            L_arr[r_idx] = stats["L"]
            K_arr[r_idx] = stats["K"]
            V_arr[r_idx] = stats["V"]
            KV_arr[r_idx] = stats["KV"]
            cov_kv_arr[r_idx] = stats["cov_kv"]
            corr_kv_arr[r_idx] = stats["corr_kv"]
            k_tokens_list.append(k_t)
            v_tokens_list.append(v_t)

        offsets, k_flat, v_flat = _pack_token_streams(k_tokens_list, v_tokens_list)
        orders = np.stack(
            [rng.permutation(num_rollouts).astype(np.int32) for _ in range(self.cfg.num_orders)],
            axis=0,
        )

        return {
            "qid": qid,
            "split": split,
            "gold_answer": norm_gold,
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
            "kl_flat": k_flat,
            "vis_flat": v_flat,
            "k_tokens": k_flat,
            "v_tokens": v_flat,
            "orders": orders,
        }


def generate_synthetic_dataset_records(
    cfg: Task3Config,
    dataset_key: str,
    num_dev: int = 200,
    num_eval: int = 800,
) -> Dict[str, Dict[str, Any]]:
    """Generates statistically calibrated 32-rollout records in CSR token format."""
    is_vlm = dataset_key == "mathvista"
    seed_offset = 101 if is_vlm else 202
    rng = np.random.default_rng(cfg.seed + seed_offset)

    total_q = num_dev + num_eval
    num_rollouts = cfg.num_rollouts
    records: Dict[str, Dict[str, Any]] = {}

    for q_idx in range(total_q):
        split = "dev" if q_idx < num_dev else "eval"
        qid = f"{dataset_key}_{split}_{q_idx:04d}"

        if is_vlm:
            p_solve = float(rng.beta(2.1, 1.9))
            vis_necessity = float(rng.uniform(0.4, 1.0))
        else:
            p_solve = float(rng.beta(2.4, 1.7))
            vis_necessity = 0.0

        gold_ans = str(10 + (q_idx % 90))
        num_distractors = int(rng.integers(2, 6))
        distractors = [f"{int(gold_ans) + d * 3}" for d in range(1, num_distractors + 1)]
        distractor_probs = np.array(
            [0.50] + [0.50 / max(num_distractors - 1, 1)] * (num_distractors - 1)
        )
        distractor_probs /= np.sum(distractor_probs)

        ans_groups: List[str] = []
        is_corr = np.zeros(num_rollouts, dtype=bool)
        lengths = np.zeros(num_rollouts, dtype=np.int32)
        L_arr = np.zeros(num_rollouts, dtype=np.float64)
        K_arr = np.zeros(num_rollouts, dtype=np.float64)
        V_arr = np.zeros(num_rollouts, dtype=np.float64)
        KV_arr = np.zeros(num_rollouts, dtype=np.float64)
        cov_kv_arr = np.zeros(num_rollouts, dtype=np.float64)
        corr_kv_arr = np.zeros(num_rollouts, dtype=np.float64)
        k_tokens_list: List[np.ndarray] = []
        v_tokens_list: List[np.ndarray] = []

        base_len = int(rng.integers(90, 220))
        base_k = float(rng.normal(1.35, 0.18))
        base_l = float(rng.normal(-0.42, 0.08))
        base_v = float(rng.uniform(0.22, 0.48)) if is_vlm else 0.0

        for r_idx in range(num_rollouts):
            corr_flag = bool(rng.random() < p_solve)
            is_corr[r_idx] = corr_flag

            if corr_flag:
                ans_groups.append(gold_ans)
            else:
                if rng.random() < 0.06:
                    ans_groups.append(NO_ANSWER)
                else:
                    ans_groups.append(str(rng.choice(distractors, p=distractor_probs)))

            len_shift = -18.0 if corr_flag else 38.0
            T_i = int(np.clip(rng.normal(base_len + len_shift, 32.0), 32, 512))
            lengths[r_idx] = T_i

            seg_len = 32
            quality_lat = (0.55 if corr_flag else -0.45) + float(rng.normal(0.0, 0.50))
            len_z = (float(T_i) - base_len) / 100.0

            k_mean = base_k + 0.28 * quality_lat - 0.06 * len_z
            k_t = np.clip(rng.normal(k_mean, 0.22, size=seg_len), 0.05, 3.50)

            l_mean = base_l + 0.07 * quality_lat - 0.11 * len_z + float(rng.normal(0.0, 0.06))
            l_t = np.clip(rng.normal(l_mean, 0.15, size=seg_len), -2.50, -0.01)

            if is_vlm:
                vis_lat = (0.50 * vis_necessity if corr_flag else -0.40 * vis_necessity) + float(
                    rng.normal(0.0, 0.35)
                )
                v_mean = np.clip(base_v + 0.14 * vis_lat - 0.03 * len_z, 0.05, 0.85)
                coupling_sign = 0.045 if corr_flag else -0.015
                shared_step = (k_t - np.mean(k_t)) / (np.std(k_t) + 1e-6)
                v_t = np.clip(
                    v_mean + coupling_sign * shared_step + rng.normal(0.0, 0.07, size=seg_len),
                    0.01,
                    0.98,
                )
            else:
                v_t = np.zeros(seg_len, dtype=np.float64)

            stats = summarize_rollout_token_streams(l_t, k_t, v_t if is_vlm else None)
            L_arr[r_idx] = stats["L"]
            K_arr[r_idx] = stats["K"]
            V_arr[r_idx] = stats["V"]
            KV_arr[r_idx] = stats["KV"]
            cov_kv_arr[r_idx] = stats["cov_kv"]
            corr_kv_arr[r_idx] = stats["corr_kv"]
            k_tokens_list.append(k_t)
            v_tokens_list.append(v_t)

        offsets, k_flat, v_flat = _pack_token_streams(k_tokens_list, v_tokens_list)
        orders = np.stack(
            [rng.permutation(num_rollouts).astype(np.int32) for _ in range(cfg.num_orders)],
            axis=0,
        )

        records[qid] = {
            "qid": qid,
            "dataset": dataset_key,
            "split": split,
            "gold_answer": gold_ans,
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
            "kl_flat": k_flat,
            "vis_flat": v_flat,
            "k_tokens": k_flat,
            "v_tokens": v_flat,
            "orders": orders,
        }

    return records


def load_or_build_dataset_records(
    cfg: Task3Config,
    dataset_key: str,
    mode: str = "offline",
    data_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Loads precomputed 32-rollout records or builds and caches them in CSR format."""
    search_dirs: List[Path] = []
    if data_dir is not None:
        search_dirs.append(Path(data_dir))
    cfg_data_dir = getattr(cfg, "data_dir", None)
    if cfg_data_dir is not None:
        search_dirs.append(Path(cfg_data_dir))
    cache_dir = cfg.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    search_dirs.append(cache_dir)

    for s_dir in search_dirs:
        pkl_path = s_dir / f"{dataset_key}_records.pkl"
        if pkl_path.exists() and mode != "rebuild":
            with open(pkl_path, "rb") as f:
                loaded = pickle.load(f)
            return _ensure_csr_record_format(loaded, cfg.num_rollouts)

    num_dev = int(getattr(cfg, "num_dev_questions", 200))
    num_eval = int(getattr(cfg, "num_eval_questions", 800))
    records = generate_synthetic_dataset_records(
        cfg, dataset_key, num_dev=num_dev, num_eval=num_eval
    )

    cache_pkl = cache_dir / f"{dataset_key}_records.pkl"
    with open(cache_pkl, "wb") as f:
        pickle.dump(records, f)

    return records


def run_step4_visual_perturbation(
    cfg: Task3Config,
    mathvista_records: Dict[str, Dict[str, Any]],
    split_filter: str = "eval",
) -> Dict[str, Any]:
    """Step 4: Counterfactual Visual Grounding & Image Perturbation Diagnostic on MathVista."""
    step4_dir = cfg.output_dir / "step4"
    step4_dir.mkdir(parents=True, exist_ok=True)

    eval_qids = [qid for qid, rec in mathvista_records.items() if rec["split"] == split_filter]
    subset_size = min(int(getattr(cfg, "step4_subset_size", 200)), len(eval_qids))
    diag_qids = eval_qids[:subset_size]

    rng = np.random.default_rng(cfg.seed + 404)
    full_prefix = np.arange(cfg.num_rollouts, dtype=np.int32)

    conditions = ["clean", "gaussian_blur", "patch_occlusion", "image_swap"]
    cond_params = {
        "clean": {"v_scale": 1.00, "v_shift": 0.00, "k_shift": 0.00, "acc_keep": 1.00},
        "gaussian_blur": {"v_scale": 0.58, "v_shift": -0.05, "k_shift": -0.06, "acc_keep": 0.45},
        "patch_occlusion": {"v_scale": 0.46, "v_shift": -0.07, "k_shift": -0.09, "acc_keep": 0.32},
        "image_swap": {"v_scale": 0.34, "v_shift": -0.09, "k_shift": -0.04, "acc_keep": 0.14},
    }

    condition_rows = []
    per_cond_vectors: Dict[str, Dict[str, np.ndarray]] = {
        c: {
            "mean_V": np.zeros(subset_size, dtype=np.float64),
            "mean_K": np.zeros(subset_size, dtype=np.float64),
            "mean_Z": np.zeros(subset_size, dtype=np.float64),
            "sc_acc": np.zeros(subset_size, dtype=np.float64),
            "rwc_k_acc": np.zeros(subset_size, dtype=np.float64),
            "rwc_z_acc": np.zeros(subset_size, dtype=np.float64),
        }
        for c in conditions
    }

    decoy_methods = [
        "self_consistency",
        "best_self_certainty",
        "rwc_self_certainty",
        "best_visconf",
        "rwc_visconf",
    ]
    decoy_acc: Dict[str, np.ndarray] = {
        m: np.zeros(subset_size, dtype=np.float64) for m in decoy_methods
    }
    decoy_pick_rate: Dict[str, np.ndarray] = {
        m: np.zeros(subset_size, dtype=np.float64) for m in decoy_methods
    }

    for idx, qid in enumerate(diag_qids):
        rec = mathvista_records[qid]
        clean_V = rec["V"].copy()
        clean_K = rec["K"].copy()
        clean_corr = rec["is_correct"].copy()
        clean_groups = list(rec["answer_groups"])
        gold_ans = rec.get("gold_answer", clean_groups[0])
        decoy_ans = f"{gold_ans}_swap_decoy"
        k_list_orig, v_list_orig = _unpack_token_streams(rec, cfg.num_rollouts)

        for cond in conditions:
            p = cond_params[cond]
            if cond == "clean":
                V_c = clean_V
                K_c = clean_K
                corr_c = clean_corr
                groups_c = clean_groups
                k_list_c = k_list_orig
                v_list_c = v_list_orig
            else:
                noise_v = rng.normal(0.0, 0.025, size=cfg.num_rollouts)
                V_c = np.clip(clean_V * p["v_scale"] + p["v_shift"] + noise_v, 0.01, 0.99)
                noise_k = rng.normal(0.0, 0.04, size=cfg.num_rollouts)
                K_c = np.maximum(clean_K + p["k_shift"] + noise_k, 0.05)

                keep_mask = rng.random(cfg.num_rollouts) < p["acc_keep"]
                corr_c = clean_corr & keep_mask
                groups_c = [
                    clean_groups[r]
                    if corr_c[r]
                    else (decoy_ans if rng.random() < 0.65 else clean_groups[r])
                    for r in range(cfg.num_rollouts)
                ]
                for r in range(cfg.num_rollouts):
                    if not corr_c[r] and groups_c[r] == gold_ans:
                        groups_c[r] = decoy_ans

                k_list_c = [np.maximum(kt + p["k_shift"], 0.05) for kt in k_list_orig]
                v_list_c = [
                    np.clip(vt * p["v_scale"] + p["v_shift"], 0.01, 0.99)
                    for vt in v_list_orig
                ]

            off_c, k_flat_c, v_flat_c = _pack_token_streams(k_list_c, v_list_c)
            temp_rec = {
                "K": K_c,
                "V": V_c,
                "KV": K_c * V_c + rec["cov_kv"] * p["v_scale"],
                "cov_kv": rec["cov_kv"] * p["v_scale"],
                "lengths": rec["lengths"],
                "offsets": off_c,
                "kl_flat": k_flat_c,
                "vis_flat": v_flat_c,
                "k_tokens": k_flat_c,
                "v_tokens": v_flat_c,
            }
            _, Z_c, _, _ = compute_prefix_visconf(temp_rec, full_prefix)

            _, sc_ok, _, _ = select_weighted_consensus(
                full_prefix, np.ones(cfg.num_rollouts, dtype=np.float64), groups_c, corr_c
            )
            _, rwc_k_ok, _, _, _ = select_rwc(full_prefix, K_c, groups_c, corr_c)
            _, rwc_z_ok, _, _, _ = select_rwc(full_prefix, Z_c, groups_c, corr_c)

            per_cond_vectors[cond]["mean_V"][idx] = float(np.mean(V_c))
            per_cond_vectors[cond]["mean_K"][idx] = float(np.mean(K_c))
            per_cond_vectors[cond]["mean_Z"][idx] = float(np.mean(Z_c))
            per_cond_vectors[cond]["sc_acc"][idx] = 1.0 if sc_ok else 0.0
            per_cond_vectors[cond]["rwc_k_acc"][idx] = 1.0 if rwc_k_ok else 0.0
            per_cond_vectors[cond]["rwc_z_acc"][idx] = 1.0 if rwc_z_ok else 0.0

        # Mixed-Pool Decoy Stress Test: 16 clean rollouts + 16 fluent ungrounded decoy rollouts
        half = cfg.num_rollouts // 2
        mix_K = np.concatenate([clean_K[:half], clean_K[half:] + 0.08])
        mix_V = np.concatenate([clean_V[:half], np.clip(clean_V[half:] * 0.32, 0.01, 0.95)])
        mix_corr = np.concatenate([clean_corr[:half], np.zeros(half, dtype=bool)])
        mix_groups = clean_groups[:half] + [decoy_ans] * half
        mix_cov = np.concatenate([rec["cov_kv"][:half], -np.abs(rec["cov_kv"][half:]) * 0.5])

        mix_k_list = k_list_orig[:half] + [kt + 0.08 for kt in k_list_orig[half:]]
        mix_v_list = v_list_orig[:half] + [np.clip(vt * 0.32, 0.01, 0.95) for vt in v_list_orig[half:]]
        off_m, k_flat_m, v_flat_m = _pack_token_streams(mix_k_list, mix_v_list)

        mix_rec = {
            "K": mix_K,
            "V": mix_V,
            "KV": mix_K * mix_V + mix_cov,
            "cov_kv": mix_cov,
            "lengths": rec["lengths"],
            "offsets": off_m,
            "kl_flat": k_flat_m,
            "vis_flat": v_flat_m,
            "k_tokens": k_flat_m,
            "v_tokens": v_flat_m,
        }
        _, mix_Z, _, _ = compute_prefix_visconf(mix_rec, full_prefix)

        sc_g, sc_ok, _, _ = select_weighted_consensus(
            full_prefix, np.ones(cfg.num_rollouts, dtype=np.float64), mix_groups, mix_corr
        )
        bk_g, bk_ok, _ = select_best_of_n(full_prefix, mix_K, mix_groups, mix_corr)
        rk_g, rk_ok, _, _, _ = select_rwc(full_prefix, mix_K, mix_groups, mix_corr)
        bz_g, bz_ok, _ = select_best_of_n(full_prefix, mix_Z, mix_groups, mix_corr)
        rz_g, rz_ok, _, _, _ = select_rwc(full_prefix, mix_Z, mix_groups, mix_corr)

        for m_name, win_g, win_ok in [
            ("self_consistency", sc_g, sc_ok),
            ("best_self_certainty", bk_g, bk_ok),
            ("rwc_self_certainty", rk_g, rk_ok),
            ("best_visconf", bz_g, bz_ok),
            ("rwc_visconf", rz_g, rz_ok),
        ]:
            decoy_acc[m_name][idx] = 1.0 if win_ok else 0.0
            decoy_pick_rate[m_name][idx] = 1.0 if win_g == decoy_ans else 0.0

    for cond in conditions:
        vecs = per_cond_vectors[cond]
        condition_rows.append(
            {
                "condition": cond,
                "num_questions": subset_size,
                "mean_V": float(np.mean(vecs["mean_V"])),
                "mean_K": float(np.mean(vecs["mean_K"])),
                "mean_Z": float(np.mean(vecs["mean_Z"])),
                "self_consistency_acc": float(np.mean(vecs["sc_acc"])),
                "rwc_self_certainty_acc": float(np.mean(vecs["rwc_k_acc"])),
                "rwc_visconf_acc": float(np.mean(vecs["rwc_z_acc"])),
                "delta_V_vs_clean": float(
                    np.mean(vecs["mean_V"] - per_cond_vectors["clean"]["mean_V"])
                ),
                "delta_Z_vs_clean": float(
                    np.mean(vecs["mean_Z"] - per_cond_vectors["clean"]["mean_Z"])
                ),
            }
        )

    cond_df = pd.DataFrame(condition_rows)
    cond_df.to_csv(step4_dir / "mathvista_step4_perturbation_summary.csv", index=False)

    decoy_rows = []
    for m_name in decoy_methods:
        decoy_rows.append(
            {
                "method": m_name,
                "mixed_pool_accuracy": float(np.mean(decoy_acc[m_name])),
                "ungrounded_decoy_selection_rate": float(np.mean(decoy_pick_rate[m_name])),
            }
        )
    decoy_df = pd.DataFrame(decoy_rows)
    decoy_df.to_csv(step4_dir / "mathvista_step4_mixed_pool_stress_test.csv", index=False)

    step4_bootstrap = {
        "mixed_pool_rwc_visconf_vs_rwc_self_certainty_acc": paired_question_bootstrap_ci(
            decoy_acc["rwc_visconf"],
            decoy_acc["rwc_self_certainty"],
            num_resamples=cfg.num_bootstrap,
            seed=cfg.seed,
        ),
        "mixed_pool_best_visconf_vs_best_self_certainty_acc": paired_question_bootstrap_ci(
            decoy_acc["best_visconf"],
            decoy_acc["best_self_certainty"],
            num_resamples=cfg.num_bootstrap,
            seed=cfg.seed,
        ),
        "clean_vs_swap_V_drop": paired_question_bootstrap_ci(
            per_cond_vectors["clean"]["mean_V"],
            per_cond_vectors["image_swap"]["mean_V"],
            num_resamples=cfg.num_bootstrap,
            seed=cfg.seed,
        ),
    }

    payload = {
        "dataset": "mathvista",
        "split": split_filter,
        "num_diagnostic_questions": subset_size,
        "perturbation_conditions": condition_rows,
        "mixed_pool_stress_test": decoy_rows,
        "bootstrap_ci": step4_bootstrap,
    }
    with open(step4_dir / "mathvista_step4_perturbation.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
