import re
import math
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

NO_ANSWER_TOKEN = "NO_ANSWER"


def get_math_verify_version() -> str:
    """Return installed math-verify package version for reproducibility reporting."""
    try:
        import importlib.metadata
        return importlib.metadata.version("math-verify")
    except Exception:
        try:
            import math_verify
            return getattr(math_verify, "__version__", "installed-unknown-version")
        except Exception:
            return "fallback-symbolic-normalizer"


def is_missing_
