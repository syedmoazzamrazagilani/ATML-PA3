import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

def normalize_question_text(text: str) -> str:
    """Normalize question text for exact GSM8K test-set deduplication."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def format_sft_example(
    tokenizer: Any,
    question: str,
    response: str,
    max_seq_len: int = 2048,
) -> Optional[Dict[str, Any]]:
    """
    Format a user question and assistant response using Qwen3 non-thinking chat template.
    Computes next-token cross-entropy only on assistant-response tokens and end-of-turn token;
    masks prompt tokens with -100. Returns None if length > max_seq_len.
    """
    if not question or not question.strip() or not response or not response.strip():
        return None

    prompt_msgs = [{"role": "user", "content": question}]
    full_msgs = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]

    try:
        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = tokenizer.apply_chat_template(
            full_msgs,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except Exception:
        prompt_str = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        full_str = f"{prompt_str}{response}<|im_end|>\n"
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = tokenizer.encode(full_str, add_special_tokens=False)

    if len(full_ids) > max_seq_len:
        return None

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + list(full_ids[prompt_len:])

    # Ensure any trailing newline after <|im_end|> is masked so supervision ends at <|im_end|>
    im_end_id = getattr(tokenizer, "eos_token_id", None)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        try:
            cand = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(cand, int) and cand >= 0:
                im_end_id = cand
        except Exception:
            pass
    if im_end_id is not None and im_end_id in full_ids[prompt_len:]:
        last_eot_idx = len(full_ids) - 1 - full_ids[::-1].index(im_end_id)
        for idx in range(last_eot_idx + 1, len(labels)):
            labels[idx] = -100

    valid_response_tokens = sum(1 for x in labels[1:] if x != -100)
    if valid_response_tokens == 0:
        return None

    return {
        "input_ids": list(full_ids),
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
        "num_valid_tokens": valid_response_tokens,
    }

def select_metamath_40k(
    metamath_iterable: Any,
    gsm8k_test_questions: List[str],
    tokenizer: Any,
    target_count: int = 40000,
    max_seq_len: int = 2048,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Select first 40,000 eligible examples from MetaMathQA in source order.
    Excludes empty questions/responses, formatted length > 2048 tokens, or
    query/original_question matching a GSM8K test question after normalization.
    """
    gsm8k_norm_set = {normalize_question_text(q) for q in gsm8k_test_questions if q}
    retained = []
    stats = {
        "inspected": 0,
        "excluded_empty": 0,
        "excluded_gsm8k_overlap": 0,
        "excluded_over_length": 0,
        "retained": 0,
    }

    for idx, row in enumerate(metamath_iterable):
        stats["inspected"] += 1
        query = (row.get("query") or "").strip()
        orig_q = (row.get("original_question") or "").strip()
        response = (row.get("response") or "").strip()

        if not query or not response:
            stats["excluded_empty"] += 1
            continue

        if (normalize_question_text(query) in gsm8k_norm_set) or (
            orig_q and normalize_question_text(orig_q) in gsm8k_norm_set
        ):
            stats["excluded_gsm8k_overlap"] += 1
            continue

        formatted = format_sft_example(tokenizer, query, response, max_seq_len=max_seq_len)
        if formatted is None:
            stats["excluded_over_length"] += 1
            continue

        formatted["source_index"] = idx
        retained.append(formatted)
        stats["retained"] += 1
        if len(retained) >= target_count:
            break

    return retained, stats

def sample_activation_probe_ids(
    num_gsm8k: int = 1319,
    num_hellaswag: int = 10042,
    sample_size: int = 32,
    seed: int = 6304,
    save_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Sample 32 GSM8K test questions and 32 HellaSwag validation contexts using seed 6304."""
    rng = np.random.default_rng(seed)
    gsm8k_ids = sorted(rng.choice(num_gsm8k, size=min(sample_size, num_gsm8k), replace=False).tolist())
    hellaswag_ids = sorted(rng.choice(num_hellaswag, size=min(sample_size, num_hellaswag), replace=False).tolist())
    probe_dict = {"seed": seed, "gsm8k_test_ids": gsm8k_ids, "hellaswag_val_ids": hellaswag_ids}
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(probe_dict, f, indent=2)
    return probe_dict

def collate_sft_batch(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Right-pad microbatch of SFT examples, masking padded labels with -100."""
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attention_mask = [], [], []
    valid_tokens = 0
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
        attention_mask.append(ex["attention_mask"] + [0] * pad_len)
        valid_tokens += ex["num_valid_tokens"]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "num_valid_tokens": torch.tensor(valid_tokens, dtype=torch.long),
    }
