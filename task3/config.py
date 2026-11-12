from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class Task3Config:
    # Seed and split sizes
    seed: int = 6304
    num_dev: int = 200
    num_eval: int = 800
    num_live_math: int = 50
    num_orders: int = 10
    num_rollouts: int = 32
    num_bootstrap: int = 1000

    # Candidate budgets and adaptive checkpoints
    budgets: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    adaptive_checkpoints: Tuple[int, ...] = (4, 8, 16)
    tau_grid: Tuple[float, ...] = (0.5, 0.625, 0.75, 0.875, 1.0)
    gamma_grid: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)

    # Live generation settings (Qwen2.5-3B-Instruct pinned revision)
    math_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    math_model_revision: str = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    system_prompt: str = "You are a helpful assistant."
    cot_suffix: str = (
        "Think step by step inside <think> </think> tags, "
        "then give your final answer inside \\boxed{}."
    )
    refinement_prompt: str = (
        "Review the solution above. Identify and correct any errors; "
        "if it is correct, retain its answer. Give your final answer inside \\boxed{}."
    )
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0

    # Paths
    data_root: Path = Path("./ATML-PA3-Data")
    mathvista_meta_path: Path = Path("./testmini.json")
    output_dir: Path = Path("./outputs/task3")

    def ensure_dirs(self) -> None:
        for sub in ["splits", "cache", "step1", "step2", "step3", "step4", "step5", "step6", "figures", "tables"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
