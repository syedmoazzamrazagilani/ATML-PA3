import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .config import Task3Config
    from .live_generation import (
        load_or_build_dataset_records,
        run_step4_visual_perturbation,
    )
    from .offline_pipeline import run_all_offline_steps_for_dataset
    from .plotting_and_tables import generate_all_plots_and_tables
except ImportError:
    from config import Task3Config
    from live_generation import (
        load_or_build_dataset_records,
        run_step4_visual_perturbation,
    )
    from offline_pipeline import run_all_offline_steps_for_dataset
    from plotting_and_tables import generate_all_plots_and_tables


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 3 Repeated Sampling, VisConf RWC, & Adaptive Compute Pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write all Step 1-5 CSVs, JSONs, figures, and tables.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional directory containing cached dataset rollout .pkl files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate (e.g. gsm8k mathvista or math500 mathvista).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "rebuild", "live"],
        help="Execution mode: 'offline' uses cached/synthetic rollouts; 'rebuild' regenerates cache.",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=None,
        help="Override number of paired question-level bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override master random seed.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip Matplotlib plot generation.",
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    cfg = Task3Config()

    if args.output_dir is not None:
        cfg.output_dir = Path(args.output_dir)
    if args.num_bootstrap is not None:
        cfg.num_bootstrap = int(args.num_bootstrap)
    if args.seed is not None:
        cfg.seed = int(args.seed)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seeds(cfg.seed)

    default_datasets = getattr(cfg, "datasets", ("math500", "mathvista"))
    datasets: List[str] = list(args.datasets) if args.datasets else list(default_datasets)

    results_by_dataset: Dict[str, Dict[str, Any]] = {}
    all_records: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ds_key in datasets:
        print(f"\n[Task 3] Loading/building 32-rollout records for dataset: {ds_key} ...")
        records = load_or_build_dataset_records(
            cfg, ds_key, mode=args.mode, data_dir=args.data_dir
        )
        all_records[ds_key] = records

        print(f"[Task 3] Running Steps 1, 2, 3, and 5 for dataset: {ds_key} ...")
        ds_results = run_all_offline_steps_for_dataset(cfg, records, ds_key)
        results_by_dataset[ds_key] = ds_results

    step4_result = None
    if "mathvista" in all_records:
        print("\n[Task 3] Running Step 4 Counterfactual Visual Perturbation on MathVista ...")
        step4_result = run_step4_visual_perturbation(
            cfg, all_records["mathvista"], split_filter="eval"
        )

    artifacts_info = {}
    if not args.skip_plots:
        print("\n[Task 3] Generating figures and formatted summary tables ...")
        artifacts_info = generate_all_plots_and_tables(
            results_by_dataset, step4_result, cfg
        )

    manifest = {
        "seed": cfg.seed,
        "output_dir": str(cfg.output_dir),
        "datasets_evaluated": datasets,
        "budgets": list(cfg.budgets),
        "num_orders": cfg.num_orders,
        "num_bootstrap": cfg.num_bootstrap,
        "artifacts": artifacts_info,
    }
    manifest_path = cfg.output_dir / "task3_full_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Task 3] Complete! Manifest saved to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    main()
