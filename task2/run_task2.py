import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.seed_utils import set_all_seeds
from common.logging_utils import save_json, load_json
from task2.data_prep import build_task2_splits
from task2.feature_cache import extract_and_cache_siglip_features, CachedFeatureStore
from task2.model_vlm import SiglipQwenVLM
from task2.trainer_vlm import train_vlm_curriculum
from task2.evaluate_vlm import evaluate_final_test_suites
from task2.plots_task2 import plot_task2_learning_curves, plot_task2_final_grounding_comparison

QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"


def load_qwen_fp16(device: str = "cuda"):
    try:
        m = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, dtype=torch.float16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=torch.float16)
    return m.to(device=device, dtype=torch.float16)


def make_synthetic_smoke_data(out_dir: Path):
    """Build synthetic image-disjoint splits & cached SigLIP features for fast pre-flight verification."""
    rng = torch.Generator().manual_seed(6304)
    feats = {i: torch.randn((196, 768), generator=rng, dtype=torch.float16) for i in range(1, 45)}
    torch.save(feats, out_dir / "siglip_features_fp16.pt")

    train_caps = [
        {"source_type": "caption", "image_id": (i % 16) + 1, "question": "Describe this image.", "target_text": "A person riding a bicycle on a street.", "is_caption": True}
        for i in range(64)
    ]
    pres_pairs = [
        {"pair_id": f"p_{i}", "image_a_id": (i % 8) + 1, "image_b_id": (i % 8) + 9, "question": "Does this image contain bicycle?", "question_type": "presence", "source_type": "presence", "answer_a": "yes", "answer_b": "no"}
        for i in range(16)
    ]
    cnt_pairs = [
        {"pair_id": f"c_{i}", "image_a_id": (i % 8) + 1, "image_b_id": (i % 8) + 9, "question": "How many dogs can you see?", "question_type": "counting", "source_type": "counting", "answer_a": "1", "answer_b": "3"}
        for i in range(16)
    ]
    vqa_pairs = [
        {"pair_id": f"v_{i}", "image_a_id": (i % 8) + 1, "image_b_id": (i % 8) + 9, "question": "What color is the car?", "question_type": "other", "source_type": "natural_vqa", "answer_a": "red", "answer_b": "blue", "gt_answers_a": ["red"] * 10, "gt_answers_b": ["blue"] * 10}
        for i in range(16)
    ]
    dev_pres = [
        {"pair_id": f"dp_{i}", "image_a_id": 17 + i, "image_b_id": 21 + i, "question": "Does this image contain cat?", "question_type": "presence", "source_type": "presence", "answer_a": "yes", "answer_b": "no"}
        for i in range(4)
    ]
    dev_vqa = [
        {"pair_id": f"dv_{i}", "image_a_id": 17 + i, "image_b_id": 21 + i, "question": "Is it sunny?", "question_type": "yes/no", "source_type": "natural_vqa", "answer_a": "yes", "answer_b": "no", "gt_answers_a": ["yes"] * 9 + ["no"], "gt_answers_b": ["no"] * 9 + ["yes"]}
        for i in range(4)
    ]
    test_pres = [
        {"pair_id": f"tp_{i}", "image_a_id": 29 + i, "image_b_id": 35 + i, "question": "Does this image contain person?", "question_type": "presence", "source_type": "presence", "answer_a": "yes", "answer_b": "no"}
        for i in range(4)
    ]
    test_vqa = [
        {"pair_id": f"tv_{i}", "image_a_id": 29 + i, "image_b_id": 35 + i, "question": "How many people are there?", "question_type": "number", "source_type": "natural_vqa", "answer_a": "2", "answer_b": "4", "gt_answers_a": ["2"] * 10, "gt_answers_b": ["4"] * 10}
        for i in range(4)
    ]
    splits = {
        "seed": 6304,
        "train_pools": {"captions": train_caps, "presence_pairs": pres_pairs, "counting_pairs": cnt_pairs, "natural_vqa_pairs": vqa_pairs},
        "dev_suites": {"captions": train_caps[:8], "grounding_pairs": dev_pres, "natural_vqa_pairs": dev_vqa},
        "test_suites": {"grounding_test": test_pres, "natural_vqa_test": test_vqa},
    }
    save_json(splits, out_dir / "task2_splits_and_ids.json")


def main():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 2 Runner")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/task2_outputs")
    parser.add_argument("--data-root", type=str, default="/kaggle/input")
    parser.add_argument("--stage", type=str, choices=["smoke", "prep_and_cache", "train", "eval", "all"], default="smoke")
    parser.add_argument("--compress-to-49", action="store_true", help="Optional Step 7: 2x2 spatial pooling to 49 visual tokens")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(6304)

    is_smoke = args.stage == "smoke"
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    splits_path = out_dir / "task2_splits_and_ids.json"
    cache_path = out_dir / "siglip_features_fp16.pt"

    if is_smoke:
        make_synthetic_smoke_data(out_dir)

    if args.stage in ("prep_and_cache", "all") and not is_smoke:
        data_root = Path(args.data_root)
        # Auto-discover COCO 2014 and VQAv2 JSON paths inside /kaggle/input
        def find_file(pattern: str) -> Path:
            matches = list(data_root.rglob(pattern))
            assert matches, f"Could not find {pattern} under {data_root}"
            return matches[0]

        splits = build_task2_splits(
            coco_train_captions_json=find_file("*captions_train2014*.json"),
            coco_val_captions_json=find_file("*captions_val2014*.json"),
            coco_train_instances_json=find_file("*instances_train2014*.json"),
            coco_val_instances_json=find_file("*instances_val2014*.json"),
            vqa_train_questions_json=find_file("*OpenEnded_mscoco_train2014_questions*.json"),
            vqa_train_annotations_json=find_file("*mscoco_train2014_annotations*.json"),
            vqa_train_comp_pairs_json=find_file("*v2_mscoco_train2014_complementary_pairs*.json"),
            vqa_val_questions_json=find_file("*OpenEnded_mscoco_val2014_questions*.json"),
            vqa_val_annotations_json=find_file("*mscoco_val2014_annotations*.json"),
            vqa_val_comp_pairs_json=find_file("*v2_mscoco_val2014_complementary_pairs*.json"),
            output_dir=out_dir,
            seed=6304,
        )

        # Map selected train2014 and val2014 image IDs to their JPEG paths
        train_img_dirs = [p.parent for p in data_root.rglob("COCO_train2014_000000*.jpg")]
        val_img_dirs = [p.parent for p in data_root.rglob("COCO_val2014_000000*.jpg")]
        train_dir = train_img_dirs[0]
        val_dir = val_img_dirs[0]

        id_to_path = {}
        for iid in splits["image_ids"]["train2014"]:
            id_to_path[int(iid)] = train_dir / f"COCO_train2014_{int(iid):012d}.jpg"
        for iid in splits["image_ids"]["val2014_dev"] + splits["image_ids"]["val2014_test"]:
            id_to_path[int(iid)] = val_dir / f"COCO_val2014_{int(iid):012d}.jpg"

        extract_and_cache_siglip_features(id_to_path, cache_path, batch_size=64, device="cuda")
        if args.stage == "prep_and_cache":
            return

    splits = load_json(splits_path)
    features_dict = torch.load(cache_path, map_location="cpu")
    feature_store = CachedFeatureStore(features_dict, compress_to_49=args.compress_to_49)

    run_subdir = out_dir / ("run_49tokens" if args.compress_to_49 else "run_196tokens")
    run_subdir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("smoke", "train", "all"):
        qwen = load_qwen_fp16(device="cuda")
        vlm = SiglipQwenVLM(qwen, num_visual_tokens=feature_store.num_visual_tokens, seed=6304)

        # Calibrate connector output magnitude against Qwen text embeddings using training images only
        sample_train_ids = [ex["image_id"] for ex in splits["train_pools"]["captions"][:64]]
        sample_train_feats = feature_store.get_batch_features(sample_train_ids, device="cuda")
        vlm.calibrate_connector(sample_train_feats)

        res = train_vlm_curriculum(
            vlm_model=vlm,
            tokenizer=tokenizer,
            feature_store=feature_store,
            train_pools=splits["train_pools"],
            dev_suites=splits["dev_suites"],
            output_dir=run_subdir,
            seed=6304,
            is_smoke=is_smoke,
        )
        stage_bounds = (2, 4) if is_smoke else (2250, 3750)
        plot_task2_learning_curves(
            res["train_history"],
            res["dev_history"],
            run_subdir / "figures" / "task2_learning_curves.pdf",
            stage_boundaries=stage_bounds,
        )
        del vlm, qwen
        gc.collect()
        torch.cuda.empty_cache()

    if args.stage in ("smoke", "eval", "all"):
        qwen = load_qwen_fp16(device="cuda")
        vlm = SiglipQwenVLM(qwen, num_visual_tokens=feature_store.num_visual_tokens, seed=6304)
        final_u = 6 if is_smoke else 4500
        ckpt_path = run_subdir / "checkpoints" / f"stage_3_update_{final_u}_weights.pt"
        vlm.load_trainable_state_dict(torch.load(ckpt_path, map_location="cuda"))

        final_res = evaluate_final_test_suites(
            vlm_model=vlm,
            tokenizer=tokenizer,
            feature_store=feature_store,
            test_suites=splits["test_suites"],
            output_dir=run_subdir / "eval",
            batch_size=8,
            max_new_tokens=16,
            num_bootstrap=(200 if is_smoke else 4000),
            seed=6304,
            is_smoke=is_smoke,
        )
        plot_task2_final_grounding_comparison(
            final_res, run_subdir / "figures" / "task2_final_grounding.pdf"
        )
        del vlm, qwen
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
