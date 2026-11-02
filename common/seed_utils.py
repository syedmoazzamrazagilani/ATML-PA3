import os
import random
import numpy as np
import torch

DEFAULT_SEED = 6304

def set_all_seeds(seed: int = DEFAULT_SEED, deterministic: bool = True) -> np.random.Generator:
    """Fix Python, NumPy, and PyTorch RNG seeds as required by ATML PA3 (seed=6304)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)

def get_rng_state_dict() -> dict:
    """Capture full RNG state for exact checkpoint resuming."""
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state_dict(state: dict) -> None:
    """Restore full RNG state when resuming from a checkpoint."""
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and "cuda_rng" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
