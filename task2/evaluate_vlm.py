import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from common.logging_utils import save_json
from task2.data_prep import normalize_vqa_answer


def vqa_soft_consensus_score(pred_norm: str, gt_answers_norm: List[str]) -> float:
    """
    Local VQA soft consensus accuracy against 10 human answers:
        s(a_hat) = (1/10) * sum_{i=1}^{10} min(1, (1/3) * sum_{j != i} 1[a_hat == a_j])
    """
    if not gt_answers_norm:
        return 0.0
    n = len(gt_answers_norm)
    total = 0.0
    for i in range(n):
        matches_other = sum(
            1 for j in range(n) if j != i and pred_norm == gt_answers_norm[j]
        )
        total += min(1.0, matches_other / 3.0)
    return total / float(n)


@torch.no_grad()
def evaluate_development_suites(
    vlm_model: nn.Module,
    tokenizer: Any,
    feature_store: Any,
    dev_suites: Dict[str, List[Dict[str, Any]]],
    update_num: int,
    stage_num: int,
    batch_size: int = 8,
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate development suites at scheduled training checkpoints:
      - Dev caption reference loss (example-weighted cross-entropy)
      - Grounding dev & Natural VQAv2 dev reference-answer NLL (excluding termination token)
        under correct, complementary, and no-image inputs
      - Image-swap penalty: Delta_NLL = NLL(y | q, I_partner) - NLL(y | q, I_correct)
    """
    vlm_model.eval()
    device = next(vlm_model.parameters()).device
    use_cuda_amp = device.type == "cuda"

    cap_items = dev_suites["captions"][: (8 if is_smoke else len(dev_suites["captions"]))]
    cap_losses = []
    for start in range(0, len(cap_items), batch_size):
        mb = cap_items[start : start + batch_size]
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda_amp):
            per_ex = vlm_model.compute_per_example_nll(
                mb, tokenizer, feature_store, include_eot=True, condition="correct"
            )
        cap_losses.extend(per_ex)

    suite_results = {}
    for suite_key in ("grounding_pairs", "natural_vqa_pairs"):
        pairs = dev_suites[suite_key][: (4 if is_smoke else len(dev_suites[suite_key]))]
        flat_examples = []
        for p in pairs:
            q_text = p["question"]
            flat_examples.append({
                "pair_id": p["pair_id"],
                "image_id": p["image_a_id"],
                "partner_image_id": p["image_b_id"],
                "question": q_text,
                "target_text": p["answer_a"],
                "is_caption": False,
            })
            flat_examples.append({
                "pair_id": p["pair_id"],
                "image_id": p["image_b_id"],
                "partner_image_id": p["image_a_id"],
                "question": q_text,
                "target_text": p["answer_b"],
                "is_caption": False,
            })

        nll_by_cond = {}
        for cond in ("correct", "complementary", "no_image"):
            vals = []
            for start in range(0, len(flat_examples), batch_size):
                mb = flat_examples[start : start + batch_size]
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_cuda_amp
                ):
                    # Exclude termination token (<|im_end|>) for reference-answer NLL
                    per_ex = vlm_model.compute_per_example_nll(
                        mb, tokenizer, feature_store, include_eot=False, condition=cond
                    )
                vals.extend(per_ex)
            nll_by_cond[cond] = vals

        mean_correct = float(np.mean(nll_by_cond["correct"]))
        mean_comp = float(np.mean(nll_by_cond["complementary"]))
        mean_noimg = float(np.mean(nll_by_cond["no_image"]))
        delta_nll = mean_comp - mean_correct

        suite_results[suite_key] = {
            "nll_correct": mean_correct,
            "nll_complementary": mean_comp,
            "nll_no_image": mean_noimg,
            "delta_nll": delta_nll,
        }

    return {
        "update": update_num,
        "stage": stage_num,
        "dev_caption_loss": float(np.mean(cap_losses)) if cap_losses else 0.0,
        "grounding_dev": suite_results["grounding_pairs"],
        "natural_vqa_dev": suite_results["natural_vqa_pairs"],
    }


def build_fixed_mismatched_mapping(
    pairs: List[Dict[str, Any]], seed: int = 6304
) -> Dict[Tuple[Any, str], int]:
    """
    Fix a deterministic mismatched-image mapping before generation, mapping each side (A or B)
    of a pair to an image from ANOTHER pair in the same suite, excluding both images of the original pair.
    """
    rng = np.random.default_rng(seed)
    n_pairs = len(pairs)
    mapping: Dict[Tuple[Any, str], int] = {}
    for idx, p in enumerate(pairs):
        orig_imgs = {p["image_a_id"], p["image_b_id"]}
        candidates = []
        for j, other in enumerate(pairs):
            if j == idx:
                continue
            if other["image_a_id"] not in orig_imgs:
                candidates.append(other["image_a_id"])
            if other["image_b_id"] not in orig_imgs:
                candidates.append(other["image_b_id"])
        if not candidates:
            # Fallback if smoke-test has only 1 pair
            candidates = [p["image_b_id"], p["image_a_id"]]
        chosen_a = int(rng.choice(candidates))
        chosen_b = int(rng.choice(candidates))
        mapping[(p["pair_id"], "A")] = chosen_a
        mapping[(p["pair_id"], "B")] = chosen_b
    return mapping


def compute_paired_bootstrap_cis(
    pair_records_by_cond: Dict[str, List[Dict[str, Any]]],
    num_resamples: int = 4000,
    seed: int = 6304,
) -> Dict[str, Any]:
    """
    Compute 95% confidence intervals using 4,000 bootstrap resamples of complete pairs,
    matching sampled pair IDs across conditions when computing differences.
    """
    rng = np.random.default_rng(seed)
    conds = list(pair_records_by_cond.keys())
    n_pairs = len(pair_records_by_cond[conds[0]])

    # Pre-extract per-pair arrays: (n_pairs,) for mean pair score, pair_acc, mean pair NLL
    arr_acc = {
        c: np.array([r["pair_mean_acc"] for r in pair_records_by_cond[c]], dtype=np.float64)
        for c in conds
    }
    arr_pair_acc = {
        c: np.array([r["pair_both_correct"] for r in pair_records_by_cond[c]], dtype=np.float64)
        for c in conds
    }
    arr_nll = {
        c: np.array([r["pair_mean_nll"] for r in pair_records_by_cond[c]], dtype=np.float64)
        for c in conds
    }

    boot_indices = rng.integers(0, n_pairs, size=(num_resamples, n_pairs))
    ci_summary: Dict[str, Any] = {"per_condition": {}, "advantages_vs_complementary": {}}

    for c in conds:
        b_acc = arr_acc[c][boot_indices].mean(axis=1) * 100.0
        b_pacc = arr_pair_acc[c][boot_indices].mean(axis=1) * 100.0
        b_nll = arr_nll[c][boot_indices].mean(axis=1)
        ci_summary["per_condition"][c] = {
            "acc_ci95": [float(np.percentile(b_acc, 2.5)), float(np.percentile(b_acc, 97.5))],
            "pair_acc_ci95": [float(np.percentile(b_pacc, 2.5)), float(np.percentile(b_pacc, 97.5))],
            "nll_ci95": [float(np.percentile(b_nll, 2.5)), float(np.percentile(b_nll, 97.5))],
        }

    if "correct" in arr_acc and "complementary" in arr_acc:
        # Correct-image advantages over complementary swaps using matched pair indices
        diff_acc = (arr_acc["correct"] - arr_acc["complementary"])[boot_indices].mean(axis=1) * 100.0
        diff_pacc = (arr_pair_acc["correct"] - arr_pair_acc["complementary"])[boot_indices].mean(axis=1) * 100.0
        # Delta_NLL = NLL(partner) - NLL(correct) > 0 means correct image makes original answer more likely
        diff_nll = (arr_nll["complementary"] - arr_nll["correct"])[boot_indices].mean(axis=1)

        ci_summary["advantages_vs_complementary"] = {
            "acc_advantage_pp": float((arr_acc["correct"].mean() - arr_acc["complementary"].mean()) * 100.0),
            "acc_advantage_ci95": [float(np.percentile(diff_acc, 2.5)), float(np.percentile(diff_acc, 97.5))],
            "pair_acc_advantage_pp": float((arr_pair_acc["correct"].mean() - arr_pair_acc["complementary"].mean()) * 100.0),
            "pair_acc_advantage_ci95": [float(np.percentile(diff_pacc, 2.5)), float(np.percentile(diff_pacc, 97.5))],
            "delta_nll": float(arr_nll["complementary"].mean() - arr_nll["correct"].mean()),
            "delta_nll_ci95": [float(np.percentile(diff_nll, 2.5)), float(np.percentile(diff_nll, 97.5))],
        }

    return ci_summary


@torch.no_grad()
def evaluate_final_test_suites(
    vlm_model: nn.Module,
    tokenizer: Any,
    feature_store: Any,
    test_suites: Dict[str, List[Dict[str, Any]]],
    output_dir: str | Path,
    batch_size: int = 8,
    max_new_tokens: int = 16,
    num_bootstrap: int = 4000,
    seed: int = 6304,
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate both final-test suites (Annotation-derived Grounding and Natural VQAv2) across
    all four image conditions:
      - Correct
      - Complementary swap
      - Mismatched
      - No image
    Uses greedy decoding, thinking disabled, 16-new-token cap, and batch size 8.
    """
    vlm_model.eval()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(vlm_model.parameters()).device
    use_cuda_amp = device.type == "cuda"

    conditions = ("correct", "complementary", "mismatched", "no_image")
    final_results: Dict[str, Any] = {}

    for suite_name in ("grounding_test", "natural_vqa_test"):
        is_grounding = suite_name == "grounding_test"
        pairs = test_suites[suite_name][: (4 if is_smoke else len(test_suites[suite_name]))]
        mismatch_map = build_fixed_mismatched_mapping(pairs, seed=seed)

        # Save the fixed mismatched-image mapping for reproducibility
        serializable_map = {f"{k[0]}_{k[1]}": v for k, v in mismatch_map.items()}
        save_json(serializable_map, out_dir / f"{suite_name}_mismatched_mapping.json")

        suite_cond_summaries = {}
        pair_records_by_cond: Dict[str, List[Dict[str, Any]]] = {}
        all_per_example_records: Dict[str, List[Dict[str, Any]]] = {}

        for cond in conditions:
            flat_examples = []
            for p in pairs:
                q_text = p["question"]
                q_type = p.get("question_type", "other")
                flat_examples.append({
                    "pair_id": p["pair_id"],
                    "side": "A",
                    "question_type": q_type,
                    "image_id": p["image_a_id"],
                    "partner_image_id": p["image_b_id"],
                    "mismatched_image_id": mismatch_map[(p["pair_id"], "A")],
                    "question": q_text,
                    "target_text": p["answer_a"],
                    "partner_target_text": p["answer_b"],
                    "gt_answers_10": p.get("gt_answers_a", [p["answer_a"]] * 10),
                    "partner_gt_answers_10": p.get("gt_answers_b", [p["answer_b"]] * 10),
                    "is_caption": False,
                })
                flat_examples.append({
                    "pair_id": p["pair_id"],
                    "side": "B",
                    "question_type": q_type,
                    "image_id": p["image_b_id"],
                    "partner_image_id": p["image_a_id"],
                    "mismatched_image_id": mismatch_map[(p["pair_id"], "B")],
                    "question": q_text,
                    "target_text": p["answer_b"],
                    "partner_target_text": p["answer_a"],
                    "gt_answers_10": p.get("gt_answers_b", [p["answer_b"]] * 10),
                    "partner_gt_answers_10": p.get("gt_answers_a", [p["answer_a"]] * 10),
                    "is_caption": False,
                })

            # 1. Teacher-forced reference-answer NLL (excluding termination token)
            nlls = []
            for start in range(0, len(flat_examples), batch_size):
                mb = flat_examples[start : start + batch_size]
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_cuda_amp
                ):
                    mb_nlls = vlm_model.compute_per_example_nll(
                        mb, tokenizer, feature_store, include_eot=False, condition=cond
                    )
                nlls.extend(mb_nlls)

            # 2. Greedy free generation (batch_size=8, max_new_tokens=16, thinking disabled)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            preds_raw = []
            for start in range(0, len(flat_examples), batch_size):
                mb = flat_examples[start : start + batch_size]
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_cuda_amp
                ):
                    mb_preds = vlm_model.generate_batch(
                        mb,
                        tokenizer,
                        feature_store,
                        condition=cond,
                        max_new_tokens=max_new_tokens,
                    )
                preds_raw.extend(mb_preds)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_time_total = time.perf_counter() - t0
            gen_time_per_ans = gen_time_total / float(max(1, len(flat_examples)))

            # 3. Score individual predictions and group by complete pairs
            per_ex_list = []
            for ex, raw_pred, nll_val in zip(flat_examples, preds_raw, nlls):
                pred_norm = normalize_vqa_answer(raw_pred)
                canon_norm = normalize_vqa_answer(ex["target_text"])
                partner_canon_norm = normalize_vqa_answer(ex["partner_target_text"])
                is_empty = pred_norm == ""

                if is_grounding:
                    acc_score = 1.0 if (not is_empty and pred_norm == canon_norm) else 0.0
                    canon_match = acc_score == 1.0
                    follows_partner = not is_empty and (pred_norm == partner_canon_norm)
                else:
                    gt10_norm = [normalize_vqa_answer(a) for a in ex["gt_answers_10"]]
                    partner_gt10_norm = [normalize_vqa_answer(a) for a in ex["partner_gt_answers_10"]]
                    acc_score = 0.0 if is_empty else vqa_soft_consensus_score(pred_norm, gt10_norm)
                    canon_match = not is_empty and (pred_norm == canon_norm)
                    follows_partner = not is_empty and (
                        pred_norm == partner_canon_norm
                        or vqa_soft_consensus_score(pred_norm, partner_gt10_norm) >= 0.9
                    )

                rec = {
                    **ex,
                    "condition": cond,
                    "raw_prediction": raw_pred,
                    "normalized_prediction": pred_norm,
                    "normalized_canonical_target": canon_norm,
                    "normalized_partner_target": partner_canon_norm,
                    "is_empty": bool(is_empty),
                    "accuracy": float(acc_score),
                    "canonical_exact_match": bool(canon_match),
                    "follows_partner_answer": bool(follows_partner),
                    "reference_nll": float(nll_val),
                }
                per_ex_list.append(rec)

            all_per_example_records[cond] = per_ex_list

            # Aggregate pair-level metrics
            pair_records = []
            for i in range(0, len(per_ex_list), 2):
                ex_a = per_ex_list[i]
                ex_b = per_ex_list[i + 1]
                both_correct = float(ex_a["canonical_exact_match"] and ex_b["canonical_exact_match"])
                identical_ans = float(
                    ex_a["normalized_prediction"] == ex_b["normalized_prediction"]
                )
                pair_records.append({
                    "pair_id": ex_a["pair_id"],
                    "question_type": ex_a["question_type"],
                    "pair_mean_acc": 0.5 * (ex_a["accuracy"] + ex_b["accuracy"]),
                    "pair_both_correct": both_correct,
                    "pair_identical_answer": identical_ans,
                    "pair_mean_nll": 0.5 * (ex_a["reference_nll"] + ex_b["reference_nll"]),
                    "pred_a": ex_a["normalized_prediction"],
                    "pred_b": ex_b["normalized_prediction"],
                    "ref_a": ex_a["normalized_canonical_target"],
                    "ref_b": ex_b["normalized_canonical_target"],
                })
            pair_records_by_cond[cond] = pair_records

            # Breakdown by question type (presence/counting or yes_no/number/other)
            qtype_scores: Dict[str, List[float]] = {}
            for r in per_ex_list:
                qtype_scores.setdefault(r["question_type"], []).append(r["accuracy"])
            qtype_summary = {
                qt: float(np.mean(vals) * 100.0) for qt, vals in qtype_scores.items()
            }

            suite_cond_summaries[cond] = {
                "condition": cond,
                "overall_accuracy": float(np.mean([r["accuracy"] for r in per_ex_list]) * 100.0),
                "by_question_type_accuracy": qtype_summary,
                "pair_accuracy": float(np.mean([p["pair_both_correct"] for p in pair_records]) * 100.0),
                "identical_answer_rate": float(np.mean([p["pair_identical_answer"] for p in pair_records]) * 100.0),
                "partner_following_rate": float(np.mean([r["follows_partner_answer"] for r in per_ex_list]) * 100.0),
                "empty_answer_rate": float(np.mean([r["is_empty"] for r in per_ex_list]) * 100.0),
                "mean_reference_nll": float(np.mean([r["reference_nll"] for r in per_ex_list])),
                "gen_time_per_answer_ms": gen_time_per_ans * 1000.0,
            }

        bootstrap_cis = compute_paired_bootstrap_cis(
            pair_records_by_cond,
            num_resamples=(200 if is_smoke else num_bootstrap),
            seed=seed,
        )

        # Select illustrative qualitative examples using a stated deterministic rule:
        # Rule: (1) a grounded pair where Correct gets both right and Complementary flips to partner answers,
        #       (2) a language-prior failure pair where Correct and Complementary produce identical answers.
        qualitative_examples = select_illustrative_complementary_examples(
            pair_records_by_cond, all_per_example_records
        )

        final_results[suite_name] = {
            "conditions": suite_cond_summaries,
            "bootstrap_95ci": bootstrap_cis,
            "illustrative_examples": qualitative_examples,
        }
        save_json(all_per_example_records, out_dir / f"{suite_name}_per_example.json")

    save_json(final_results, out_dir / "final_test_results.json")
    return final_results


def select_illustrative_complementary_examples(
    pair_records_by_cond: Dict[str, List[Dict[str, Any]]],
    all_per_example_records: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Deterministic selection rule for qualitative complementary-pair examples:
      1. Visual Grounding Success: lowest pair_id where both images are answered correctly under
         'correct' AND both predictions flip to follow the partner image under 'complementary'.
      2. Language Prior / Consistency Failure: lowest pair_id where 'correct', 'complementary',
         and 'no_image' all output the exact same identical answer for both images.
    """
    corr_ex = all_per_example_records["correct"]
    comp_ex = all_per_example_records["complementary"]
    mism_ex = all_per_example_records["mismatched"]
    noim_ex = all_per_example_records["no_image"]

    selected = []
    for i in range(0, len(corr_ex), 2):
        ca, cb = corr_ex[i], corr_ex[i + 1]
        sa, sb = comp_ex[i], comp_ex[i + 1]
        ma, mb = mism_ex[i], mism_ex[i + 1]
        na, nb = noim_ex[i], noim_ex[i + 1]

        is_grounded_success = (
            ca["canonical_exact_match"]
            and cb["canonical_exact_match"]
            and sa["follows_partner_answer"]
            and sb["follows_partner_answer"]
        )
        is_prior_failure = (
            ca["normalized_prediction"] == cb["normalized_prediction"]
            and sa["normalized_prediction"] == sb["normalized_prediction"]
            and na["normalized_prediction"] == nb["normalized_prediction"]
        )

        if is_grounded_success or is_prior_failure or len(selected) < 4:
            category = (
                "grounded_flip_success"
                if is_grounded_success
                else ("language_prior_collapse" if is_prior_failure else "general_case")
            )
            selected.append({
                "selection_category": category,
                "pair_id": ca["pair_id"],
                "question_type": ca["question_type"],
                "question": ca["question"],
                "image_a_id": ca["image_id"],
                "image_b_id": cb["image_id"],
                "ref_a": ca["target_text"],
                "ref_b": cb["target_text"],
                "pred_correct": [ca["raw_prediction"], cb["raw_prediction"]],
                "pred_complementary": [sa["raw_prediction"], sb["raw_prediction"]],
                "pred_mismatched": [ma["raw_prediction"], mb["raw_prediction"]],
                "pred_no_image": [na["raw_prediction"], nb["raw_prediction"]],
            })
            if len(selected) >= 6:
                break
    return selected
