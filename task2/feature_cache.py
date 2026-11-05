import gc
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipVisionModel

from common.logging_utils import save_json, get_gpu_memory_mb, reset_peak_gpu_memory

VISION_MODEL_ID = "google/siglip-base-patch16-224"


def _locate_or_fetch_coco_image(
    split: str, image_id: int, local_search_dirs: List[Path]
) -> Image.Image:
    """
    Locate `COCO_{split}_{image_id:012d}.jpg` in mounted local directories,
    or fetch deterministically from `http://images.cocodataset.org/{split}/...`.
    """
    filename = f"COCO_{split}_{int(image_id):012d}.jpg"
    for d in local_search_dirs:
        cand1 = d / split / filename
        if cand1.exists():
            return Image.open(cand1).convert("RGB")
        cand2 = d / filename
        if cand2.exists():
            return Image.open(cand2).convert("RGB")

    url = f"http://images.cocodataset.org/{split}/{filename}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}")


@torch.no_grad()
def cache_siglip_patch_features(
    splits_payload: Dict[str, Any],
    output_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 32,
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Extract 196 final patch features (14x14x768) from frozen `google/siglip-base-patch16-224`.
    Note: HuggingFace `SiglipVisionModel.forward()` applies `self.vision_model.post_layernorm`
    exactly once to produce `outputs.last_hidden_state` of shape (B, 196, 768) in 14x14 spatial order.
    Saves cached features in FP16 and removes the vision encoder from GPU memory.
    """
    out_path = Path(output_dir)
    cache_file = out_path / "siglip_features_fp16.pt"
    stats_file = out_path / "feature_cache_stats.json"

    if cache_file.exists() and stats_file.exists():
        print(f"[Task 2 Cache] Loading cached SigLIP features from {cache_file} ...")
        return torch.load(cache_file, map_location="cpu"), json_load_safe(stats_file)

    out_path.mkdir(parents=True, exist_ok=True)
    reset_peak_gpu_memory()
    t0 = time.perf_counter()

    processor = AutoImageProcessor.from_pretrained(VISION_MODEL_ID)
    try:
        vision_model = SiglipVisionModel.from_pretrained(
            VISION_MODEL_ID, dtype=torch.float16
        ).to(device)
    except TypeError:
        vision_model = SiglipVisionModel.from_pretrained(
            VISION_MODEL_ID, torch_dtype=torch.float16
        ).to(device)
    vision_model.eval()
    for p in vision_model.parameters():
        p.requires_grad_(False)

    # Discover any local COCO directories under /kaggle/input
    local_dirs: List[Path] = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("train2014"):
            if p.is_dir():
                local_dirs.append(p.parent)
        for p in kaggle_input.rglob("val2014"):
            if p.is_dir() and p.parent not in local_dirs:
                local_dirs.append(p.parent)

    train_ids = splits_payload["train"]["unique_train2014_image_ids"]
    val_ids = sorted(
        set(splits_payload["dev"]["unique_val2014_image_ids"])
        | set(splits_payload["test"]["unique_val2014_image_ids"])
    )

    work_items: List[Tuple[str, int]] = (
        [("train2014", int(i)) for i in train_ids]
        + [("val2014", int(i)) for i in val_ids]
    )
    print(f"[Task 2 Cache] Caching {len(work_items)} unique COCO images in FP16 ...")

    feature_dict: Dict[str, torch.Tensor] = {}
    pool = ThreadPoolExecutor(max_workers=num_workers)

    try:
        for start in range(0, len(work_items), batch_size):
            batch_items = work_items[start : start + batch_size]
            imgs = list(
                pool.map(
                    lambda item: _locate_or_fetch_coco_image(item[0], item[1], local_dirs),
                    batch_items,
                )
            )
            inputs = processor(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float16)

            with torch.amp.autocast(
                device_type="cuda" if device == "cuda" else "cpu",
                dtype=torch.float16,
                enabled=(device == "cuda"),
            ):
                # last_hidden_state has post_layernorm applied once; shape (B, 196, 768)
                outputs = vision_model(pixel_values=pixel_values)
                patch_feats = outputs.last_hidden_state.detach().to(
                    device="cpu", dtype=torch.float16
                )

            assert patch_feats.shape[1:] == (196, 768), (
                f"Unexpected SigLIP patch shape: {patch_feats.shape}"
            )
            for idx, (split_name, img_id) in enumerate(batch_items):
                key = f"{split_name}_{img_id}"
                feature_dict[key] = patch_feats[idx].clone()

            if (start // batch_size) % 20 == 0:
                print(f"  Cached {min(len(work_items), start + batch_size)}/{len(work_items)} images ...")
    finally:
        pool.shutdown(wait=True)

    elapsed = time.perf_counter() - t0
    mem_stats = get_gpu_memory_mb()

    # Remove vision encoder from GPU memory before training Qwen
    del vision_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.save(feature_dict, cache_file)
    cache_size_mb = cache_file.stat().st_size / (1024 ** 2)
    stats = {
        "vision_encoder": VISION_MODEL_ID,
        "num_cached_images": len(feature_dict),
        "feature_shape": [196, 768],
        "dtype": "float16",
        "post_layernorm_applied_once": True,
        "caching_time_sec": elapsed,
        "caching_peak_gpu_memory_mb": mem_stats["max_allocated_mb"],
        "cache_file_size_mb": cache_size_mb,
    }
    save_json(stats, stats_file)
    print(f"[Task 2 Cache] Saved {len(feature_dict)} image features ({cache_size_mb:.1f} MB) in {elapsed:.1f}s.")
    return feature_dict, stats


def json_load_safe(path: Path) -> Dict[str, Any]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
