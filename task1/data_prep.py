import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
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


def apply_chat_template_to_ids(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> List[int]:
    """
    Safely invoke tokenizer.apply_chat_template and always return a flat List[int],
    handling both transformers 4.x (List[int]) and transformers 5.x (BatchEncoding dict).
    """
    try:
        raw = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            return_dict=False,
        )
    except TypeError:
        try:
            raw = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            # Fallback for mock tokenizers
            text = ""
            for m in messages:
                text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            if add_generation_prompt:
                text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            raw = tokenizer.encode(text, add_special_tokens=False)

    if isinstance(raw, dict) or hasattr(raw, "keys"):
        raw = raw["input_ids"]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        raw = raw[0]
    return [int(x) for x in raw]


def format_sft_example(
    tokenizer: Any,
    question: str,
    response: str,
    max_seq_len: int = 2048,
) -> Dict[str, Any] | None:
    """
    Format a user question and supplied assistant response using Qwen3's chat template
    with thinking disabled (enable_thinking=False)[cite: 3].
    Computes next-token cross-entropy only on assistant-response tokens and the
    end-of-turn token (<|im_end|>); masks prompts and trailing newlines with -100[cite: 3].
    Returns None if empty or formatted length > max_seq_len (never truncates)[cite: 3].
    """
    if not question or not question.strip() or not response or not response.strip():
        return None

    prompt_msgs = [{"role": "user", "content": question}]
    full_msgs = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]

    prompt_ids = apply_chat_template_to_ids(
        tokenizer, prompt_msgs, add_generation_prompt=True, enable_thinking=False
    )
    full_ids = apply_chat_template_to_ids(
        tokenizer, full_msgs, add_generation_prompt=False, enable_thinking=False
    )

    if len(full_ids) > max_seq_len or len(full_ids) <= len(prompt_ids):
        return None

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + list(full_ids[prompt_len:])

    # Ensure supervision stops at the end-of-turn token (<|im_end|>)[cite: 3]
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
    Select the first 40,000 eligible examples from MetaMathQA in source order[cite: 3].
    Excludes empty questions/responses, formatted length > 2048 tokens, or
    query/original_question matching any GSM8K test question after normalization[cite: 3].
    """
    gsm8k_norm_set = {normalize_question_text(q) for q in gsm8k_test_questions if q}
    retained: List[Dict[str, Any]] = []
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

        formatted = format_sft_example(
            tokenizer, query, response, max_seq_len=max_seq_len
        )
        if formatted is None:
            stats["excluded_over_length"] += 1
            continue

        formatted["source_index"] = idx
        retained.append(formatted)
        stats["retained"] += 1
        if len(retained) >= target_count:
            break

    if len(retained) < target_count:
        raise RuntimeError(
            f"Only retained {len(retained)}/{target_count} examples! Stats: {stats}"
        )

    return retained, stats


def sample_activation_probe_ids(
    num_gsm8k: int = 1319,
    num_hellaswag: int = 10042,
    sample_size: int = 32,
    seed: int = 6304,
    save_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Sample 32 GSM8K test questions and 32 HellaSwag validation contexts using seed 6304[cite: 3]."""
    rng = np.random.default_rng(seed)
    gsm8k_ids = sorted(
        rng.choice(num_gsm8k, size=min(sample_size, num_gsm8k), replace=False).tolist()
    )
    hellaswag_ids = sorted(
        rng.choice(num_hellaswag, size=min(sample_size, num_hellaswag), replace=False).tolist()
    )
    probe_dict = {
        "seed": seed,
        "gsm8k_test_ids": gsm8k_ids,
        "hellaswag_val_ids": hellaswag_ids,
    }
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(probe_dict, f, indent=2)
    return probe_dict


class SFTDataset(Dataset):
    """Order-preserving dataset of tokenized SFT examples."""
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def collate_sft_batch(
    batch: List[Dict[str, Any]], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """Right-pad microbatch of SFT examples, masking padded labels with -100[cite: 3]."""
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
