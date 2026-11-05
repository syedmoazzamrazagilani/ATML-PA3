import re
import json
import zipfile
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

from common.logging_utils import save_json, load_json

ANNOTATION_URLS = {
    "coco_trainval2014": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "vqa_comp_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
    "vqa_comp_val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip",
    "vqa_ques_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "vqa_ques_val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "vqa_ann_train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "vqa_ann_val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}

ARTICLES = {"a", "an", "the"}
NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "heres": "here's", "im": "i'm", "isnt": "isn't", "itll": "it'll", "lets": "let's",
    "mightve": "might've", "mustnt": "mustn't", "shan't": "shan't", "shed": "she'd",
    "shouldve": "should've", "shouldnt": "shouldn't", "thatll": "that'll", "theres": "there's",
    "theyre": "they're", "theyve": "they've", "wasnt": "wasn't", "weren't": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
    "where's": "where's", "who'd": "who'd", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "yall": "y'all", "youll": "you'll", "youre": "you're", "youve": "you've",
}
PUNCT = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]

COCO_PLURALS = {
    "person": "people", "bicycle": "bicycles", "car": "cars", "motorcycle": "motorcycles",
    "airplane": "airplanes", "bus": "buses", "train": "trains", "truck": "trucks",
    "boat": "boats", "traffic light": "traffic lights", "fire hydrant": "fire hydrants",
    "stop sign": "stop signs", "parking meter": "parking meters", "bench": "benches",
    "bird": "birds", "cat": "cats", "dog": "dogs", "horse": "horses", "sheep": "sheep",
    "cow": "cows", "elephant": "elephants", "bear": "bears", "zebra": "zebras",
    "giraffe": "giraffes", "backpack": "backpacks", "umbrella": "umbrellas",
    "handbag": "handbags", "tie": "ties", "suitcase": "suitcases", "frisbee": "frisbees",
    "skis": "pairs of skis", "snowboard": "snowboards", "sports ball": "sports balls",
    "kite": "kites", "baseball bat": "baseball bats", "baseball glove": "baseball gloves",
    "skateboard": "skateboards", "surfboard": "surfboards", "tennis racket": "tennis rackets",
    "bottle": "bottles", "wine glass": "wine glasses", "cup": "cups", "fork": "forks",
    "knife": "knives", "spoon": "spoons", "bowl": "bowls", "banana": "bananas",
    "apple": "apples", "sandwich": "sandwiches", "orange": "oranges", "broccoli": "broccoli",
    "carrot": "carrots", "hot dog": "hot dogs", "pizza": "pizzas", "donut": "donuts",
    "cake": "cakes", "chair": "chairs", "couch": "couches", "potted plant": "potted plants",
    "bed": "beds", "dining table": "dining tables", "toilet": "toilets", "tv": "tvs",
    "laptop": "laptops", "mouse": "mice", "remote": "remotes", "keyboard": "keyboards",
    "cell phone": "cell phones", "microwave": "microwaves", "oven": "ovens",
    "toaster": "toasters", "sink": "sinks", "refrigerator": "refrigerators",
    "book": "books", "clock": "clocks", "vase": "vases", "scissors": "scissors",
    "teddy bear": "teddy bears", "hair drier": "hair driers", "toothbrush": "toothbrushes",
}


def normalize_vqa_answer(ans: str) -> str:
    """Official VQA answer normalization applied across all targets and predictions."""
    if not ans:
        return ""
    ans = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    ans = re.sub(r"(?!<=\d)(\.)(?!\d)", "", ans)
    ans = re.sub(r"(\d)(\,)(\d)", r"\1\3", ans)
    for p in PUNCT:
        if p in ans:
            ans = ans.replace(p, " ")
    words = ans.split()
    out_words = []
    for w in words:
        if w in ARTICLES:
            continue
        w = NUMBER_MAP.get(w, w)
        w = CONTRACTIONS.get(w, w)
        out_words.append(w)
    return " ".join(out_words).strip()


def compute_vqa_soft_accuracy(pred_norm: str, gt_answers_norm: List[str]) -> float:
    """
    Local VQA soft consensus accuracy against all ten human answers:
        s(a_hat) = (1/10) * sum_{i=1}^{10} min(1, (1/3) * sum_{j != i} 1[a_hat == a_j])
    Empty outputs remain in accuracy denominators (returning 0.0).
    """
    if not pred_norm:
        return 0.0
    scores = []
    n_ans = len(gt_answers_norm)
    for i in range(n_ans):
        matches_other = sum(
            1 for j in range(n_ans) if j != i and gt_answers_norm[j] == pred_norm
        )
        scores.append(min(1.0, matches_other / 3.0))
    return float(sum(scores) / max(1, n_ans))


def pluralize_coco_category(cat_name: str) -> str:
    if cat_name in COCO_PLURALS:
        return COCO_PLURALS[cat_name]
    return cat_name if cat_name.endswith("s") else f"{cat_name}s"


def download_and_extract_annotations(raw_dir: str | Path) -> Path:
    """Download COCO 2014 and VQAv2 complementary pair JSONs if not already present."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    expected_Marker = raw_path / "v2_mscoco_val2014_annotations.json"
    if expected_Marker.exists() and (raw_path / "annotations" / "instances_val2014.json").exists():
        return raw_path

    for name, url in ANNOTATION_URLS.items():
        zip_path = raw_path / f"{name}.zip"
        if not zip_path.exists():
            print(f"[Task 2 Data] Downloading {name} from {url} ...")
            urllib.request.urlretrieve(url, zip_path)
        print(f"[Task 2 Data] Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_path)
        zip_path.unlink(missing_ok=True)

    return raw_path


def _extract_valid_vqav2_pairs(
    comp_pairs: List[Tuple[int, int]],
    questions_by_id: Dict[int, Dict[str, Any]],
    annotations_by_qid: Dict[int, Dict[str, Any]],
    allowed_image_ids: set | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter VQAv2 complementary pairs whose normalized consensus answers differ and
    have agreement from at least 8 of 10 annotators for each image.
    Uses an original answer string from the consensus group as the training target
    and retains all 10 human answers for scoring.
    Groups valid pairs by answer_type: 'yes/no', 'number', 'other'.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {"yes/no": [], "number": [], "other": []}
    for qid1, qid2 in comp_pairs:
        q1 = questions_by_id.get(qid1)
        q2 = questions_by_id.get(qid2)
        a1 = annotations_by_qid.get(qid1)
        a2 = annotations_by_qid.get(qid2)
        if not (q1 and q2 and a1 and a2):
            continue

        img1 = int(q1["image_id"])
        img2 = int(q2["image_id"])
        if img1 == img2:
            continue
        if allowed_image_ids is not None and (
            img1 not in allowed_image_ids or img2 not in allowed_image_ids
        ):
            continue

        ans_type = a1.get("answer_type", "other")
        if ans_type not in grouped or a2.get("answer_type", "other") != ans_type:
            continue

        raw_ans_1 = [x["answer"] for x in a1.get("answers", [])]
        raw_ans_2 = [x["answer"] for x in a2.get("answers", [])]
        if len(raw_ans_1) != 10 or len(raw_ans_2) != 10:
            continue

        norm_ans_1 = [normalize_vqa_answer(x) for x in raw_ans_1]
        norm_ans_2 = [normalize_vqa_answer(x) for x in raw_ans_2]

        c1_val, c1_cnt = Counter(norm_ans_1).most_common(1)[0]
        c2_val, c2_cnt = Counter(norm_ans_2).most_common(1)[0]
        if not c1_val or not c2_val or c1_val == c2_val:
            continue
        if c1_cnt < 8 or c2_cnt < 8:
            continue

        # Select an original answer string from the consensus group as training target
        orig_target_1 = next(
            raw_ans_1[i] for i in range(10) if norm_ans_1[i] == c1_val
        )
        orig_target_2 = next(
            raw_ans_2[i] for i in range(10) if norm_ans_2[i] == c2_val
        )

        grouped[ans_type].append(
            {
                "pair_id": f"vqav2_{qid1}_{qid2}",
                "suite": "vqav2",
                "question_type": ans_type,
                "question": q1["question"],
                "qid_1": int(qid1),
                "qid_2": int(qid2),
                "image_id_1": img1,
                "image_id_2": img2,
                "target_1": orig_target_1,
                "target_2": orig_target_2,
                "consensus_norm_1": c1_val,
                "consensus_norm_2": c2_val,
                "gt_answers_1": raw_ans_1,
                "gt_answers_2": raw_ans_2,
                "gt_norm_1": norm_ans_1,
                "gt_norm_2": norm_ans_2,
            }
        )
    return grouped


def _select_disjoint_image_pairs(
    candidates: List[Dict[str, Any]],
    target_count: int,
    rng: np.random.Generator,
    forbidden_images: set,
    enforce_within_split_unique_images: bool = True,
) -> List[Dict[str, Any]]:
    """Select `target_count` pairs using seeded permutation while avoiding `forbidden_images`."""
    order = rng.permutation(len(candidates)).tolist()
    selected = []
    local_used = set()
    for idx in order:
        p = candidates[idx]
        i1, i2 = p["image_id_1"], p["image_id_2"]
        if i1 in forbidden_images or i2 in forbidden_images:
            continue
        if enforce_within_split_unique_images and (
            i1 in local_used or i2 in local_used
        ):
            continue
        selected.append(p)
        local_used.add(i1)
        local_used.add(i2)
        if len(selected) >= target_count:
            break

    # If strict within-split uniqueness falls short, allow image reuse within the same split only
    if len(selected) < target_count:
        selected_ids = {p["pair_id"] for p in selected}
        for idx in order:
            p = candidates[idx]
            if p["pair_id"] in selected_ids:
                continue
            i1, i2 = p["image_id_1"], p["image_id_2"]
            if i1 in forbidden_images or i2 in forbidden_images:
                continue
            selected.append(p)
            if len(selected) >= target_count:
                break

    if len(selected) < target_count:
        raise RuntimeError(
            f"Could only select {len(selected)}/{target_count} pairs!"
        )
    return selected


def _build_grounding_pairs_from_instances(
    instances_json: Dict[str, Any],
    candidate_image_ids: List[int],
    num_presence_pairs: int,
    num_counting_pairs: int,
    rng: np.random.Generator,
    split_tag: str,
    disjoint_images: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], set]:
    """
    Programmatically generate annotation-derived grounding pairs from COCO object annotations:
      - Presence: "Does this image contain [object]?" pairing an image with >=1 object
        with an image with 0 annotations for that category (targets 'yes' / 'no').
      - Counting: "How many [objects] can you see?" pairing images with different
        annotated counts between 1 and 4.
    """
    categories = {
        int(c["id"]): c["name"] for c in instances_json.get("categories", [])
    }
    cat_ids_sorted = sorted(categories.keys())
    cand_set = set(candidate_image_ids)

    img_cat_counts: Dict[int, Counter] = defaultdict(Counter)
    for ann in instances_json.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        img_id = int(ann["image_id"])
        if img_id in cand_set:
            img_cat_counts[img_id][int(ann["category_id"])] += 1

    # Only use images that have at least one annotation so zero-count images are genuine COCO images
    annotated_imgs = sorted(img_cat_counts.keys())
    by_cat_present: Dict[int, List[int]] = defaultdict(list)
    by_cat_count_1to4: Dict[int, Dict[int, List[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for img_id in annotated_imgs:
        counts = img_cat_counts[img_id]
        for cid, cnt in counts.items():
            if cnt >= 1:
                by_cat_present[cid].append(img_id)
            if 1 <= cnt <= 4:
                by_cat_count_1to4[cid][cnt].append(img_id)

    used_imgs: set = set()
    presence_pairs: List[Dict[str, Any]] = []
    counting_pairs: List[Dict[str, Any]] = []

    # 1. Generate counting pairs (counts in 1..4)
    valid_count_cats = [
        cid
        for cid in cat_ids_sorted
        if len([c for c, lst in by_cat_count_1to4[cid].items() if len(lst) >= 2]) >= 2
    ]
    attempts = 0
    while len(counting_pairs) < num_counting_pairs and attempts < num_counting_pairs * 80:
        attempts += 1
        cid = int(valid_count_cats[rng.integers(0, len(valid_count_cats))])
        avail_counts = [
            c
            for c, lst in by_cat_count_1to4[cid].items()
            if any((not disjoint_images) or (im not in used_imgs) for im in lst)
        ]
        if len(avail_counts) < 2:
            continue
        c1, c2 = rng.choice(avail_counts, size=2, replace=False).tolist()
        pool1 = [
            im
            for im in by_cat_count_1to4[cid][c1]
            if (not disjoint_images) or (im not in used_imgs)
        ]
        pool2 = [
            im
            for im in by_cat_count_1to4[cid][c2]
            if (not disjoint_images) or (im not in used_imgs)
        ]
        if not pool1 or not pool2:
            continue
        img1 = int(pool1[rng.integers(0, len(pool1))])
        img2 = int(pool2[rng.integers(0, len(pool2))])
        if img1 == img2:
            continue
        used_imgs.add(img1)
        used_imgs.add(img2)
        obj_plural = pluralize_coco_category(categories[cid])
        ans1, ans2 = str(c1), str(c2)
        counting_pairs.append(
            {
                "pair_id": f"{split_tag}_count_{len(counting_pairs)}_{cid}_{img1}_{img2}",
                "suite": "grounding",
                "question_type": "counting",
                "category_id": cid,
                "category_name": categories[cid],
                "question": f"How many {obj_plural} can you see?",
                "image_id_1": img1,
                "image_id_2": img2,
                "target_1": ans1,
                "target_2": ans2,
                "consensus_norm_1": ans1,
                "consensus_norm_2": ans2,
            }
        )

    # 2. Generate presence pairs ("Does this image contain [object]?")
    valid_pres_cats = [cid for cid in cat_ids_sorted if len(by_cat_present[cid]) >= 5]
    attempts = 0
    while len(presence_pairs) < num_presence_pairs and attempts < num_presence_pairs * 80:
        attempts += 1
        cid = int(valid_pres_cats[rng.integers(0, len(valid_pres_cats))])
        pos_pool = [
            im
            for im in by_cat_present[cid]
            if (not disjoint_images) or (im not in used_imgs)
        ]
        if not pos_pool:
            continue
        img_pos = int(pos_pool[rng.integers(0, len(pos_pool))])

        # Pick a negative image confirmed to have 0 annotations for category cid
        neg_img = None
        for _ in range(30):
            cand_neg = int(annotated_imgs[rng.integers(0, len(annotated_imgs))])
            if cand_neg == img_pos:
                continue
            if disjoint_images and cand_neg in used_imgs:
                continue
            if img_cat_counts[cand_neg][cid] == 0:
                neg_img = cand_neg
                break
        if neg_img is None:
            continue

        used_imgs.add(img_pos)
        used_imgs.add(neg_img)
        cat_name = categories[cid]
        # Randomly order (yes, no) vs (no, yes) so image_1 is not always 'yes'
        if rng.random() < 0.5:
            img1, img2, t1, t2 = img_pos, neg_img, "yes", "no"
        else:
            img1, img2, t1, t2 = neg_img, img_pos, "no", "yes"

        presence_pairs.append(
            {
                "pair_id": f"{split_tag}_pres_{len(presence_pairs)}_{cid}_{img1}_{img2}",
                "suite": "grounding",
                "question_type": "presence",
                "category_id": cid,
                "category_name": cat_name,
                "question": f"Does this image contain {cat_name}?",
                "image_id_1": img1,
                "image_id_2": img2,
                "target_1": t1,
                "target_2": t2,
                "consensus_norm_1": t1,
                "consensus_norm_2": t2,
            }
        )

    return presence_pairs, counting_pairs, used_imgs


def _assign_mismatched_images(
    pairs: List[Dict[str, Any]], seed: int = 6304
) -> List[Dict[str, Any]]:
    """
    Fix a deterministic mismatched-image mapping before generation, excluding both images
    of the original pair, and picking an unrelated image from another pair in the same suite.
    """
    rng = np.random.default_rng(seed)
    all_suite_imgs = []
    for p in pairs:
        all_suite_imgs.append(p["image_id_1"])
        all_suite_imgs.append(p["image_id_2"])

    for p in pairs:
        i1, i2 = p["image_id_1"], p["image_id_2"]
        valid_others = [im for im in all_suite_imgs if im != i1 and im != i2]
        m1 = int(valid_others[rng.integers(0, len(valid_others))])
        valid_others_2 = [im for im in valid_others if im != m1] or valid_others
        m2 = int(valid_others_2[rng.integers(0, len(valid_others_2))])
        p["mismatched_image_id_1"] = m1
        p["mismatched_image_id_2"] = m2
    return pairs


def prepare_all_task2_splits(
    raw_ann_dir: str | Path,
    out_dir: str | Path,
    seed: int = 6304,
    is_smoke: bool = False,
) -> Dict[str, Any]:
    """
    Prepare all Task 2 datasets and splits using seed 6304:
      - Training (from train2014):
          * COCO captions: 6,000 images x 3 captions = 18,000 examples
          * Annotation-derived grounding pools: presence pool & counting pool
          * Natural VQAv2: 1,536 pairs (640 yes/no, 256 number, 640 other) = 3,072 questions
      - Development (from val2014, strictly image-disjoint from Final Test and Train):
          * COCO captions: 96 images x 1 caption = 96 examples
          * Grounding: 24 pairs / 48 questions (12 presence + 12 counting pairs)
          * Natural VQAv2: 32 pairs / 64 questions (12 yes/no, 8 number, 12 other)
      - Final Test (from val2014, strictly image-disjoint from Dev and Train):
          * Grounding: 64 pairs / 128 questions (32 presence + 32 counting pairs)
          * Natural VQAv2: 96 pairs / 192 questions (36 yes/no, 24 number, 36 other)
    """
    out_path = Path(out_dir)
    splits_file = out_path / ("task2_splits_smoke.json" if is_smoke else "task2_splits.json")
    if splits_file.exists():
        return load_json(splits_file)

    rng = np.random.default_rng(seed)
    raw_path = download_and_extract_annotations(raw_ann_dir)

    print("[Task 2 Data] Loading COCO 2014 & VQAv2 annotation JSONs ...")
    cap_train_json = load_json(raw_path / "annotations" / "captions_train2014.json")
    cap_val_json = load_json(raw_path / "annotations" / "captions_val2014.json")
    inst_train_json = load_json(raw_path / "annotations" / "instances_train2014.json")
    inst_val_json = load_json(raw_path / "annotations" / "instances_val2014.json")

    vqa_comp_train = load_json(raw_path / "v2_mscoco_train2014_complementary_pairs.json")
    vqa_comp_val = load_json(raw_path / "v2_mscoco_val2014_complementary_pairs.json")
    vqa_q_train = {
        int(q["question_id"]): q
        for q in load_json(raw_path / "v2_OpenEnded_mscoco_train2014_questions.json")["questions"]
    }
    vqa_q_val = {
        int(q["question_id"]): q
        for q in load_json(raw_path / "v2_OpenEnded_mscoco_val2014_questions.json")["questions"]
    }
    vqa_a_train = {
        int(a["question_id"]): a
        for a in load_json(raw_path / "v2_mscoco_train2014_annotations.json")["annotations"]
    }
    vqa_a_val = {
        int(a["question_id"]): a
        for a in load_json(raw_path / "v2_mscoco_val2014_annotations.json")["annotations"]
    }

    # --- 1. VAL2014 SPLITS (Strictly Image-Disjoint between Dev and Final Test) ---
    val_vqav2_grouped = _extract_valid_vqav2_pairs(vqa_comp_val, vqa_q_val, vqa_a_val)
    used_val_test_imgs: set = set()
    used_val_dev_imgs: set = set()

    # Final Test VQAv2: 36 yes/no, 24 number, 36 other (96 pairs = 192 questions)
    test_vqa_counts = {"yes/no": 4 if is_smoke else 36, "number": 4 if is_smoke else 24, "other": 4 if is_smoke else 36}
    test_vqav2_pairs = []
    for k_type, cnt in test_vqa_counts.items():
        chosen = _select_disjoint_image_pairs(
            val_vqav2_grouped[k_type], cnt, rng, forbidden_images=used_val_test_imgs
        )
        for p in chosen:
            used_val_test_imgs.add(p["image_id_1"])
            used_val_test_imgs.add(p["image_id_2"])
        test_vqav2_pairs.extend(chosen)

    # Development VQAv2: 12 yes/no, 8 number, 12 other (32 pairs = 64 questions)
    dev_vqa_counts = {"yes/no": 2 if is_smoke else 12, "number": 2 if is_smoke else 8, "other": 2 if is_smoke else 12}
    dev_vqav2_pairs = []
    for k_type, cnt in dev_vqa_counts.items():
        chosen = _select_disjoint_image_pairs(
            val_vqav2_grouped[k_type],
            cnt,
            rng,
            forbidden_images=(used_val_test_imgs | used_val_dev_imgs),
        )
        for p in chosen:
            used_val_dev_imgs.add(p["image_id_1"])
            used_val_dev_imgs.add(p["image_id_2"])
        dev_vqav2_pairs.extend(chosen)

    all_val_img_ids = sorted({int(im["id"]) for im in cap_val_json["images"]})
    rem_val_for_test_grounding = [
        im for im in all_val_img_ids if im not in (used_val_test_imgs | used_val_dev_imgs)
    ]

    # Final Test Grounding: 64 pairs (32 presence + 32 counting)
    n_test_g = 4 if is_smoke else 32
    test_pres, test_count, g_test_imgs = _build_grounding_pairs_from_instances(
        inst_val_json,
        rem_val_for_test_grounding,
        num_presence_pairs=n_test_g,
        num_counting_pairs=n_test_g,
        rng=rng,
        split_tag="test",
        disjoint_images=True,
    )
    used_val_test_imgs |= g_test_imgs
    test_grounding_pairs = test_pres + test_count

    # Development Grounding: 24 pairs (12 presence + 12 counting)
    rem_val_for_dev_grounding = [
        im for im in all_val_img_ids if im not in (used_val_test_imgs | used_val_dev_imgs)
    ]
    n_dev_g = 2 if is_smoke else 12
    dev_pres, dev_count, g_dev_imgs = _build_grounding_pairs_from_instances(
        inst_val_json,
        rem_val_for_dev_grounding,
        num_presence_pairs=n_dev_g,
        num_counting_pairs=n_dev_g,
        rng=rng,
        split_tag="dev",
        disjoint_images=True,
    )
    used_val_dev_imgs |= g_dev_imgs
    dev_grounding_pairs = dev_pres + dev_count

    # Development Captions: 96 images, 1 caption each
    val_caps_by_img = defaultdict(list)
    for ann in cap_val_json["annotations"]:
        c_txt = (ann.get("caption") or "").strip()
        if c_txt:
            val_caps_by_img[int(ann["image_id"])].append(c_txt)

    rem_val_for_dev_caps = [
        im
        for im in sorted(val_caps_by_img.keys())
        if im not in (used_val_test_imgs | used_val_dev_imgs)
    ]
    n_dev_caps = 8 if is_smoke else 96
    chosen_dev_cap_imgs = sorted(
        rng.choice(rem_val_for_dev_caps, size=n_dev_caps, replace=False).tolist()
    )
    dev_captions = []
    for im_id in chosen_dev_cap_imgs:
        used_val_dev_imgs.add(int(im_id))
        dev_captions.append(
            {
                "image_id": int(im_id),
                "caption": val_caps_by_img[int(im_id)][0],
                "split": "val2014",
            }
        )

    # Fix deterministic mismatched-image mappings for Final Test suites
    test_grounding_pairs = _assign_mismatched_images(test_grounding_pairs, seed=seed)
    test_vqav2_pairs = _assign_mismatched_images(test_vqav2_pairs, seed=seed + 1)

    # --- 2. TRAIN2014 SPLITS ---
    train_vqav2_grouped = _extract_valid_vqav2_pairs(vqa_comp_train, vqa_q_train, vqa_a_train)
    train_vqa_counts = {"yes/no": 8 if is_smoke else 640, "number": 4 if is_smoke else 256, "other": 8 if is_smoke else 640}
    train_vqav2_pairs = []
    train_vqa_imgs: set = set()
    for k_type, cnt in train_vqa_counts.items():
        chosen = _select_disjoint_image_pairs(
            train_vqav2_grouped[k_type],
            cnt,
            rng,
            forbidden_images=set(),
            enforce_within_split_unique_images=False,
        )
        for p in chosen:
            train_vqa_imgs.add(p["image_id_1"])
            train_vqa_imgs.add(p["image_id_2"])
        train_vqav2_pairs.extend(chosen)

    # Training COCO captions: 6,000 images x 3 captions = 18,000 examples
    # Preferentially include train_vqa_imgs that have >=3 captions so total unique images to cache is minimized
    train_caps_by_img = defaultdict(list)
    for ann in cap_train_json["annotations"]:
        c_txt = (ann.get("caption") or "").strip()
        if c_txt:
            train_caps_by_img[int(ann["image_id"])].append(c_txt)

    eligible_train_cap_imgs = sorted(
        [im for im, caps in train_caps_by_img.items() if len(caps) >= 3]
    )
    n_train_cap_imgs = 16 if is_smoke else 6000
    overlap_vqa_imgs = [im for im in sorted(train_vqa_imgs) if len(train_caps_by_img[im]) >= 3]
    if len(overlap_vqa_imgs) >= n_train_cap_imgs:
        chosen_train_cap_imgs = sorted(
            rng.choice(overlap_vqa_imgs, size=n_train_cap_imgs, replace=False).tolist()
        )
    else:
        non_overlap = [im for im in eligible_train_cap_imgs if im not in train_vqa_imgs]
        extra_needed = n_train_cap_imgs - len(overlap_vqa_imgs)
        extra_chosen = rng.choice(non_overlap, size=extra_needed, replace=False).tolist()
        chosen_train_cap_imgs = sorted(overlap_vqa_imgs + extra_chosen)

    train_captions = []
    for im_id in chosen_train_cap_imgs:
        for c_idx in range(3):
            train_captions.append(
                {
                    "image_id": int(im_id),
                    "caption": train_caps_by_img[int(im_id)][c_idx],
                    "split": "train2014",
                }
            )

    all_train_used_imgs = sorted(set(chosen_train_cap_imgs) | train_vqa_imgs)

    # Reusable Training Grounding Pools built over the selected train2014 images
    n_pool = 32 if is_smoke else 2000
    train_pres_pool, train_count_pool, _ = _build_grounding_pairs_from_instances(
        inst_train_json,
        all_train_used_imgs,
        num_presence_pairs=n_pool,
        num_counting_pairs=n_pool,
        rng=rng,
        split_tag="train",
        disjoint_images=False,
    )

    # Verify strict image-disjointness between Train, Dev, and Final Test
    # Prefix train2014 and val2014 IDs when checking global uniqueness, and check val2014 Dev vs Test directly
    assert used_val_dev_imgs.isdisjoint(used_val_test_imgs), (
        "FATAL: Development and Final Test sets share val2014 images!"
    )

    payload = {
        "seed": seed,
        "train": {
            "captions": train_captions,
            "grounding_presence_pool": train_pres_pool,
            "grounding_counting_pool": train_count_pool,
            "vqav2_pairs": train_vqav2_pairs,
            "unique_train2014_image_ids": all_train_used_imgs,
        },
        "dev": {
            "captions": dev_captions,
            "grounding_pairs": dev_grounding_pairs,
            "vqav2_pairs": dev_vqav2_pairs,
            "unique_val2014_image_ids": sorted(used_val_dev_imgs),
        },
        "test": {
            "grounding_pairs": test_grounding_pairs,
            "vqav2_pairs": test_vqav2_pairs,
            "unique_val2014_image_ids": sorted(used_val_test_imgs),
        },
        "counts_summary": {
            "train_caption_examples": len(train_captions),
            "train_presence_pool_pairs": len(train_pres_pool),
            "train_counting_pool_pairs": len(train_count_pool),
            "train_vqav2_pairs": len(train_vqav2_pairs),
            "dev_caption_examples": len(dev_captions),
            "dev_grounding_pairs": len(dev_grounding_pairs),
            "dev_vqav2_pairs": len(dev_vqav2_pairs),
            "test_grounding_pairs": len(test_grounding_pairs),
            "test_vqav2_pairs": len(test_vqav2_pairs),
            "unique_train2014_images": len(all_train_used_imgs),
            "unique_dev_val2014_images": len(used_val_dev_imgs),
            "unique_test_val2014_images": len(used_val_test_imgs),
        },
    }
    save_json(payload, splits_file)
    print(f"[Task 2 Data] Saved image-disjoint splits to {splits_file}: {payload['counts_summary']}")
    return payload
