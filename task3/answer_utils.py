import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NO_ANSWER = "NO_ANSWER"

try:
    import importlib.metadata
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify

    MATH_VERIFY_VERSION = importlib.metadata.version("math-verify")
    HAS_MATH_VERIFY = True
except Exception:
    MATH_VERIFY_VERSION = "fallback-0.0"
    HAS_MATH_VERIFY = False


def extract_last_boxed(text: Optional[str]) -> Optional[str]:
    """
    Finds the last literal '\\boxed{' and scans balanced braces,
    preserving nested LaTeX and stripping surrounding whitespace.
    Absent, empty, or unclosed last box returns None.
    """
    if not text or not isinstance(text, str):
        return None
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    n = len(text)
    while i < n and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    content = text[start : i - 1].strip()
    return content if len(content) > 0 else None


def _clean_basic_latex(s: str) -> str:
    s = s.strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = re.sub(r"\\(?:mathrm|text|textbf|displaystyle)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace(" ", "")
    return s


def parse_math_expr(ans: Optional[str]) -> Any:
    """Parses a raw MATH answer string using math-verify."""
    if ans is None or not isinstance(ans, str) or ans.strip() == "":
        return None
    raw = ans.strip()
    if not HAS_MATH_VERIFY:
        return _clean_basic_latex(raw)
    try:
        # Wrap in $...$ so LatexExtractionConfig reliably parses raw boxed contents like \frac{1}{2}
        wrapped = raw if (raw.startswith("$") and raw.endswith("$")) else f"${raw}$"
        parsed = parse(
            wrapped,
            extraction_config=[
                LatexExtractionConfig(),
                ExprExtractionConfig(),
                StringExtractionConfig(),
            ],
        )
        if parsed:
            return parsed
        return parse(raw)
    except Exception:
        return None


def math_answers_equivalent(ans_a: Optional[str], parsed_a: Any, ans_b: Optional[str], parsed_b: Any) -> bool:
    """Checks mathematical equivalence between two non-null answers using math-verify."""
    if ans_a is None or ans_b is None:
        return False
    clean_a = _clean_basic_latex(ans_a)
    clean_b = _clean_basic_latex(ans_b)
    if clean_a == clean_b and clean_a != "":
        return True
    if not HAS_MATH_VERIFY:
        return False
    if not parsed_a or not parsed_b:
        return False
    try:
        return bool(verify(parsed_a, parsed_b))
    except Exception:
        return False


def group_and_grade_math_question(
    extracted_answers: List[Optional[str]], gold_answer: str
) -> Tuple[List[str], List[bool]]:
    """
    Forms answer equivalence groups for the 32 rollouts of a single MATH question
    WITHOUT consulting gold_answer, then grades each group using the same rules.
    Returns (group_labels, is_correct_list).
    """
    gold_str = str(gold_answer).strip() if gold_answer is not None else ""
    # Extract boxed from gold if gold_answer contains \boxed{}
    gold_boxed = extract_last_boxed(gold_str)
    if gold_boxed is not None:
        gold_str = gold_boxed
    gold_parsed = parse_math_expr(gold_str)

    # Deduplicate raw strings to minimize math-verify calls
    unique_raw: List[str] = []
    raw_to_uidx: Dict[str, int] = {}
    for ans in extracted_answers:
        if ans is not None and isinstance(ans, str) and ans.strip() != "":
            s = ans.strip()
            if s not in raw_to_uidx:
                raw_to_uidx[s] = len(unique_raw)
                unique_raw.append(s)

    parsed_unique = [parse_math_expr(s) for s in unique_raw]

    # Form equivalence groups among unique non-null answers in order of appearance
    rep_indices: List[int] = []
    rep_labels: List[str] = []
    uidx_to_group: Dict[int, str] = {}

    for uidx, s in enumerate(unique_raw):
        p_u = parsed_unique[uidx]
        assigned_label = None
        for r_pos, r_uidx in enumerate(rep_indices):
            if math_answers_equivalent(unique_raw[r_uidx], parsed_unique[r_uidx], s, p_u):
                assigned_label = rep_labels[r_pos]
                break
        if assigned_label is None:
            assigned_label = _clean_basic_latex(s) or f"GROUP_{len(rep_indices)}"
            # Ensure distinct group labels for mathematically distinct representatives
            if assigned_label in rep_labels or assigned_label == NO_ANSWER:
                assigned_label = f"{assigned_label}__g{len(rep_indices)}"
            rep_indices.append(uidx)
            rep_labels.append(assigned_label)
        uidx_to_group[uidx] = assigned_label

    # Grade each unique answer against gold_answer using the exact same equivalence rule
    uidx_to_correct: Dict[int, bool] = {}
    for uidx, s in enumerate(unique_raw):
        uidx_to_correct[uidx] = math_answers_equivalent(gold_str, gold_parsed, s, parsed_unique[uidx])

    group_labels: List[str] = []
    is_correct: List[bool] = []
    for ans in extracted_answers:
        if ans is None or not isinstance(ans, str) or ans.strip() == "":
            group_labels.append(NO_ANSWER)
            is_correct.append(False)
        else:
            u = raw_to_uidx[ans.strip()]
            group_labels.append(uidx_to_group[u])
            is_correct.append(uidx_to_correct[u])

    return group_labels, is_correct


def _try_parse_float(val_str: str) -> Optional[float]:
    s = val_str.strip().replace(",", "")
    if s.endswith("%"):
         s = s[:-1].strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Check simple fraction a/b or \frac{a}{b}
    m_frac = re.match(r"^\\frac\{([+-]?\d+(?:\.\d+)?)\}\{([+-]?\d+(?:\.\d+)?)\}$", s)
    if m_frac:
        denom = float(m_frac.group(2))
        return float(m_frac.group(1)) / denom if denom != 0 else None
    m_slash = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)$", s)
    if m_slash:
        denom = float(m_slash.group(2))
        return float(m_slash.group(1)) / denom if denom != 0 else None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def normalize_mathvista_answer(ans: Optional[str], meta: Dict[str, Any]) -> str:
    """
    Normalizes a MathVista answer using question metadata (question_type, answer_type,
    choices, precision) from testmini.json.
    Multiple-choice answers are represented as uppercase option letters (A, B, C, ...).
    Missing extractions map to NO_ANSWER.
    """
    if ans is None or not isinstance(ans, str) or ans.strip() == "":
        return NO_ANSWER
    s = ans.strip()
    if s.upper() == NO_ANSWER:
        return NO_ANSWER

    q_type = meta.get("question_type", "free_form")
    a_type = meta.get("answer_type", "text")
    choices = meta.get("choices") or []
    precision = meta.get("precision", None)

    if q_type == "multi_choice" and len(choices) > 0:
        valid_letters = [chr(ord("A") + i) for i in range(len(choices))]
        # 1. Exact option letter match: "A", "(B)", "C.", "Option D", "Choice A"
        m_letter = re.match(r"^(?:option|choice|\()?([A-Za-z])(?:\)|\.|:)?$", s, flags=re.IGNORECASE)
        if m_letter:
            cand_letter = m_letter.group(1).upper()
            if cand_letter in valid_letters:
                return cand_letter

        # 2. Exact match against choice strings (case-insensitive / stripped)
        s_clean = s.strip().lower()
        for idx, ch_val in enumerate(choices):
            if str(ch_val).strip().lower() == s_clean:
                return valid_letters[idx]

        # 3. Numeric match against choice values if answer_type is integer/float or numeric
        s_num = _try_parse_float(s)
        if s_num is not None:
            for idx, ch_val in enumerate(choices):
                ch_num = _try_parse_float(str(ch_val))
                if ch_num is not None and abs(s_num - ch_num) <= 1e-6:
                    return valid_letters[idx]

        # 4. Leading option letter pattern like "A) 12" or "(B) 45"
        m_prefix = re.match(r"^\(?([A-Za-z])[\)\.\:]\s+.*$", s)
        if m_prefix and m_prefix.group(1).upper() in valid_letters:
            return m_prefix.group(1).upper()

        return f"MC_OTHER::{s_clean}"

    # Free-form rules by answer_type
    if a_type == "integer":
        v = _try_parse_float(s)
        if v is not None:
            return str(int(round(v)))
    elif a_type == "float":
        v = _try_parse_float(s)
        if v is not None:
            prec = int(precision) if precision is not None else 2
            prec = max(0, min(prec, 8))
            rounded = round(v, prec)
            if abs(rounded) < 10 ** (-prec - 1):
                rounded = 0.0
            return f"{rounded:.{prec}f}"
    elif a_type == "list":
        # Normalize brackets and numeric elements
        inner = s.strip("[]() ")
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        norm_parts = []
        for p in parts:
            vp = _try_parse_float(p)
            if vp is not None:
                norm_parts.append(str(int(round(vp))) if abs(vp - round(vp)) < 1e-6 else f"{vp:.4g}")
            else:
                norm_parts.append(p.lower())
        return "[" + ", ".join(norm_parts) + "]"

    # Default text / fallback numeric normalization
    v = _try_parse_float(s)
    if v is not None:
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return f"{round(v, 4):.4g}"
    return _clean_basic_latex(s).lower()


def group_and_grade_mathvista_question(
    extracted_answers: List[Optional[str]],
    gold_answer: str,
    meta: Dict[str, Any],
) -> Tuple[List[str], List[bool]]:
    """
    Normalizes and groups MathVista answers using testmini.json metadata without
    consulting gold_answer during grouping, then grades against normalized gold_answer.
    """
    # Use metadata gold answer if present, else stored gold_answer
    gold_raw = str(gold_answer) if gold_answer is not None else str(meta.get("answer", ""))
    norm_gold = normalize_mathvista_answer(gold_raw, meta)
    # Also normalize meta['answer'] if stored gold_answer was already an option letter or raw value
    if "answer" in meta and meta["answer"] is not None:
        norm_meta_gold = normalize_mathvista_answer(str(meta["answer"]), meta)
    else:
        norm_meta_gold = norm_gold

    group_labels: List[str] = []
    is_correct: List[bool] = []
    for ans in extracted_answers:
        norm_pred = normalize_mathvista_answer(ans, meta)
        group_labels.append(norm_pred)
        if norm_pred == NO_ANSWER:
            is_correct.append(False)
        else:
            corr = (norm_pred == norm_gold) or (norm_pred == norm_meta_gold)
            is_correct.append(bool(corr))
    return group_labels, is_correct
