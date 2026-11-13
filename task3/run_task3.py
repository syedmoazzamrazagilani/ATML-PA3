import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from .config import Task3Config
    from .live_generation import (
        extract_step6_qualitative_cases,
        load_or_build_dataset_records,
        run_step4_live_math_experiment,
    )
    from .offline_pipeline import run_all_offline_steps_for_dataset
    from .plotting_and_tables import generate_all_plots_and_tables
except ImportError:
    from config import Task3Config
    from live_generation import (
        extract_step6_qualitative_cases,
        load_or_build_dataset_records,
        run_step4_live_math_experiment,
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
        description="Run Task 3: Test-Time Compute (Offline Rollout Analysis + Live Generation)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["smoke", "offline", "live", "all"],
        help="Execution stage: 'smoke' (fast verification), 'offline' (Steps 1,2,3,5), 'live' (Step 4), or 'all'.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/kaggle/working/data/task3"),
        help="Directory containing ATML-PA3-Data parquet folders and testmini.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/task3_outputs"),
        help="Directory to write all Step 1-6 CSVs, JSONs, figures, and tables.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset identifiers to evaluate (default: math mathvista).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "rebuild"],
        help="Cache mode for parquet record construction.",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=None,
        help="Override number of paired question-level bootstrap resamples (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override master random seed (default: 6304).",
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    cfg = Task3Config()

    cfg.output_dir = Path(args.output_dir)
    if args.seed is not None:
        cfg.seed = int(args.seed)

    is_smoke = args.stage == "smoke"
    if is_smoke:
        cfg.num_bootstrap = 20
    elif args.num_bootstrap is not None:
        cfg.num_bootstrap = int(args.num_bootstrap)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seeds(cfg.seed)

    default_datasets = getattr(cfg, "datasets", ("math", "mathvista"))
    datasets: List[str] = list(args.datasets) if args.datasets else list(default_datasets)

    results_by_dataset: Dict[str, Dict[str, Any]] = {}
    all_records: Dict[str, Dict[str, Dict[str, Any]]] = {}

    max_q = 24 if is_smoke else None

    for ds_key in datasets:
        print(f"\n[Task 3] Loading supplied rollouts for dataset: {ds_key} ...")
        records = load_or_build_dataset_records(
            cfg,
            ds_key,
            mode=args.mode,
            data_dir=args.data_root,
            max_questions=max_q,
        )
        all_records[ds_key] = records

        if args.stage in ("smoke", "offline", "all"):
            print(f"[Task 3] Running offline Steps 1, 2, 3, and 5 for: {ds_key} ...")
            ds_results = run_all_offline_steps_for_dataset(cfg, records, ds_key)
            results_by_dataset[ds_key] = ds_results

    step4_result = None
    math_key = next((k for k in all_records.keys() if k.lower() != "mathvista"), None)
    if args.stage in ("smoke", "live", "all") and math_key is not None:
        print(f"\n[Task 3] Running Step 4 Live Generation on {math_key} ...")
        step4_result = run_step4_live_math_experiment(
            cfg,
            all_records[math_key],
            data_dir=args.data_root,
            num_questions=50,
            smoke_mode=is_smoke,
        )

    step6_cases = extract_step6_qualitative_cases(cfg, all_records, step4_result)

    artifacts_info = {}
    if results_by_dataset:
        print("\n[Task 3] Generating figures and formatted summary tables ...")
        artifacts_info = generate_all_plots_and_tables(
            results_by_dataset, step4_result, cfg
        )

    manifest = {
        "stage": args.stage,
        "seed": cfg.seed,
        "output_dir": str(cfg.output_dir),
        "datasets_evaluated": datasets,
        "budgets": list(cfg.budgets),
        "num_orders": cfg.num_orders,
        "num_bootstrap": cfg.num_bootstrap,
        "step4_completed": step4_result is not None,
        "step6_cases_extracted": bool(step6_cases),
        "artifacts": artifacts_info,
    }
    manifest_path = cfg.output_dir / "task3_full_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Task 3] Stage '{args.stage}' complete! Manifest saved to: {manifest_path}")
    return manifest


if __name__ == "__main__":
    main()
