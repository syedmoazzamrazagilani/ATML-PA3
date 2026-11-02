import json
import time
from pathlib import Path
from typing import Any, Dict
import torch

class JsonlLogger:
    """Append-only JSONL logger for training steps, evaluations, and diagnostics."""
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        record_with_ts = {"timestamp": time.time(), **record}
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record_with_ts) + "\n")

def save_json(data: Any, filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: str | Path) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_gpu_memory_mb() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "max_allocated_mb": 0.0, "reserved_mb": 0.0}
    torch.cuda.synchronize()
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
    }

def reset_peak_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
