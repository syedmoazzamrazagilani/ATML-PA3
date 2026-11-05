import gc
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_utils import save_json, get_gpu_memory_mb, reset_peak_gpu_memory

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"


class CachedFeatureStore:
    """
    Memory-mapped / RAM-loaded store of cached FP16 SigLIP patch features (196 x 768).
    Supports optional 2x2 spatial average pooling to 49 x 768 visual tokens (Step 7).
    """
    def __init__(self, features_dict: Dict[int, torch.Tensor], compress_to_49: bool = False):
        self.features_dict = features_dict
        self.compress_to_49 = compress_to_49

    @property
    def num_visual_tokens(self) -> int:
        return 49 if self.compress_to_49 else 196

    def get_batch_features(self, image_ids: List[int], device: torch.device | str) -> torch.Tensor:
        """
        Return batch of visual features of shape (B, 196, 768) or (B, 49, 768) in FP16.
        For 49-token compression, reshapes 196 -> (14, 14, 768) and averages each 2x2 spatial block.
        """
        feats = torch.stack([self.features_dict[int(iid)] for iid in image_ids], dim=0).to(
            device=device, dtype=torch.float16, non_blocking=True
        )
        if self.compress_to_49 and feats.size(1) == 196:
            B, _, D = feats.shape
            grid = feats.view(B, 14, 14, D).permute(0, 3, 1, 2)  # (B, 768, 14, 14)
            pooled = F.avg_pool2d(grid.float(), kernel_size=2, stride=2).to(torch.float16)
            feats = pooled.permute(0, 2, 3, 1).reshape(B, 49, D)
        return feats


@torch.no_grad()
def extract_and_cache_siglip_features(
    image_id_to_filepath: Dict[int, str | Path],
    cache_path: str | Path,
    batch_size: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Extract the 196 final patch features (14 x 14 spatial order, 768-dim) from frozen
    google/siglip-base-patch16-224 with final layer normalization (`post_layernorm`)
    applied exactly once. Saves in FP16 and removes the vision encoder from GPU memory.
    """
    from transformers import AutoImageProcessor, SiglipVisionModel

    out_path = Path(cache_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reset_peak_gpu_memory()
    t0 = time.perf_counter()

    processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_ID)
    try:
        vision_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_ID, dtype=torch.float16)
    except TypeError:
        vision_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_ID, torch_dtype=torch.float16)
    vision_model = vision_model.to(device).eval()
    for p in vision_model.parameters():
        p.requires_grad_(False)

    sorted_ids = sorted(image_id_to_filepath.keys())
    cached_features: Dict[int, torch.Tensor] = {}

    for start in range(0, len(sorted_ids), batch_size):
        batch_ids = sorted_ids[start : start + batch_size]
        pil_images = []
        for iid in batch_ids:
            with Image.open(image_id_to_filepath[iid]) as im:
                pil_images.append(im.convert("RGB"))
        inputs = processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float16)

        # SiglipVisionModel.forward applies encoder + post_layernorm exactly once on last_hidden_state
        outputs = vision_model(pixel_values=pixel_values, interpolate_pos_encoding=False)
        patch_feats = outputs.last_hidden_state.detach().to(device="cpu", dtype=torch.float16)
        assert patch_feats.shape[1:] == (196, 768), f"Unexpected SigLIP shape: {patch_feats.shape}"

        for iid, feat in zip(batch_ids, patch_feats):
            cached_features[int(iid)] = feat.clone()

    torch.save(cached_features, out_path)
    elapsed_sec = time.perf_counter() - t0
    mem_stats = get_gpu_memory_mb()

    # Explicitly remove the vision encoder from GPU memory before training Qwen
    del vision_model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = {
        "num_cached_images": len(cached_features),
        "feature_shape": [196, 768],
        "cache_time_sec": elapsed_sec,
        "peak_gpu_memory_mb": mem_stats["max_allocated_mb"],
        "cache_file_size_mb": out_path.stat().st_size / (1024 ** 2),
    }
    save_json(stats, out_path.with_suffix(".stats.json"))
    return stats
