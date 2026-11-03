from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

from common.logging_utils import save_json


def patch_tokenizer_non_thinking(tokenizer: Any) -> Any:
    """
    Ensure tokenizer.apply_chat_template defaults to enable_thinking=False[cite: 3] and
    returns a flat List[int] (not a BatchEncoding dict) when tokenize=True in transformers 5.x.
    """
    if getattr(tokenizer, "_is_patched_non_thinking", False):
        return tokenizer

    orig_fn = tokenizer.apply_chat_template

    def wrapped_apply_chat_template(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        want_dict = kwargs.get("return_dict", False)
        if kwargs.get("tokenize", False) and not want_dict:
            kwargs["return_dict"] = False
        res = orig_fn(*args, **kwargs)
        if kwargs.get("tokenize", False) and not want_dict:
            if isinstance(res, dict) or hasattr(res, "keys"):
                res = res["input_ids"]
        return res

    tokenizer.apply_chat_template = wrapped_apply_chat_template
    tokenizer._is_patched_non_thinking = True
    return tokenizer


def run_lm_eval_for_condition(
    model: nn.Module,
    tokenizer: Any,
    condition_name: str,
    output_dir: str | Path,
    batch_size: int = 8,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model condition on GSM8K (generation, non-thinking chat template, greedy, 1024 cap)[cite: 3]
    and HellaSwag, BoolQ, ARC-Easy (candidate likelihood, stock completion prompts, no chat template)[cite: 3].
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    out_dir = Path(output_dir) / condition_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(torch.float16)
    patch_tokenizer_non_thinking(tokenizer)

    hflm = HFLM(
        pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=2048
    )

    # 1. GSM8K zero-shot generation with non-thinking chat template[cite: 3]
    gsm8k_res = lm_eval.simple_evaluate(
        model=hflm,
        tasks=["gsm8k_cot"],
        num_fewshot=0,
        apply_chat_template=True,
        gen_kwargs="do_sample=False,max_gen_toks=1024",
        log_samples=True,
        limit=limit,
    )

    # 2. Likelihood benchmarks without chat template[cite: 3]
    likelihood_res = lm_eval.simple_evaluate(
        model=hflm,
        tasks=["hellaswag", "boolq", "arc_easy"],
        num_fewshot=0,
        apply_chat_template=False,
        log_samples=True,
        limit=limit,
    )

    g_metrics = gsm8k_res["results"].get("gsm8k_cot", {})
    h_metrics = likelihood_res["results"].get("hellaswag", {})
    b_metrics = likelihood_res["results"].get("boolq", {})
    a_metrics = likelihood_res["results"].get("arc_easy", {})

    gsm8k_samples = gsm8k_res.get("samples", {}).get("gsm8k_cot", [])
    per_example_gsm8k = []
    truncated_count = 0
    unextractable_count = 0

    for idx, s in enumerate(gsm8k_samples):
        resp_text = ""
        if s.get("resps") and s["resps"][0]:
            resp_text = s["resps"][0][0]
        gen_tok_len = (
            len(tokenizer.encode(resp_text, add_special_tokens=False))
            if resp_text
            else 0
        )
        is_truncated = gen_tok_len >= 1024
        if is_truncated:
            truncated_count += 1
        filtered_ans = s.get("filtered_resps", [None])[0]
        is_unextractable = filtered_ans in (None, "", "[invalid]")
        if is_unextractable:
            unextractable_count += 1
        per_example_gsm8k.append(
            {
                "example_idx": idx,
                "target": s.get("target"),
                "extracted_answer": filtered_ans,
                "exact_match": s.get("exact_match", 0.0),
                "gen_token_len": gen_tok_len,
                "truncated": is_truncated,
                "unextractable": is_unextractable,
                "response_text": resp_text,
            }
        )

    n_gsm = max(1, len(per_example_gsm8k))
    summary = {
        "condition": condition_name,
        "gsm8k_flex_em": float(
            g_metrics.get(
                "exact_match,flexible-extract", g_metrics.get("exact_match,none", 0.0)
            )
        )
        * 100.0,
        "gsm8k_strict_em": float(g_metrics.get("exact_match,strict-match", 0.0)) * 100.0,
        "gsm8k_truncation_rate": (truncated_count / n_gsm) * 100.0,
        "gsm8k_unextractable_rate": (unextractable_count / n_gsm) * 100.0,
        "hellaswag_acc": float(h_metrics.get("acc,none", 0.0)) * 100.0,
        "hellaswag_acc_norm": float(h_metrics.get("acc_norm,none", 0.0)) * 100.0,
        "boolq_acc": float(b_metrics.get("acc,none", 0.0)) * 100.0,
        "arc_easy_acc": float(a_metrics.get("acc,none", 0.0)) * 100.0,
        "arc_easy_acc_norm": float(a_metrics.get("acc_norm,none", 0.0)) * 100.0,
    }

    save_json(summary, out_dir / "benchmark_summary.json")
    save_json(per_example_gsm8k, out_dir / "gsm8k_per_example.json")
    for task_name in ("hellaswag", "boolq", "arc_easy"):
        task_samples = likelihood_res.get("samples", {}).get(task_name, [])
        save_json(task_samples, out_dir / f"{task_name}_per_example.json")

    return summary


def compute_benchmark_deltas_table(
    summaries: List[Dict[str, Any]],
    baseline_condition: str = "baseline",
) -> List[Dict[str, Any]]:
    """Compute percentage-point changes (Delta pp) relative to the shared baseline condition[cite: 4, 5]."""
    base = next(
        (s for s in summaries if s["condition"] == baseline_condition), summaries[0]
    )
    keys = [
        "gsm8k_flex_em",
        "hellaswag_acc",
        "hellaswag_acc_norm",
        "boolq_acc",
        "arc_easy_acc",
        "arc_easy_acc_norm",
    ]
    rows = []
    for s in summaries:
        row = dict(s)
        for k in keys:
            row[f"{k}_delta_pp"] = float(s[k] - base[k])
        rows.append(row)
    return rows
