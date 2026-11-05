import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from collections import Counter, defaultdict
import numpy as np

from common.logging_utils import save_json

ARTICLES = {"a", "an", "the"}
NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hell": "he'll", "hes": "he's", "id": "i'd", "ill": "i'll", "im": "i'm",
    "ive": "i've", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "its": "it's",
    "lets": "let's", "mightve": "might've", "mustve": "must've", "shan't": "shan't",
    "shed": "she'd", "shell": "she'll", "shes": "she's", "shouldve": "should've",
    "shouldnt": "couldn't", "thatll": "that'll", "thats": "that's", "there's": "there's",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "wasnt": "wasn't", "wed": "we'd", "well": "we'll", "were": "we're", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "where's": "where's", "whod": "who'd", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "yall": "y'all", "youd": "you'd", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}
PLURAL_MAP = {
    "person": "people", "man": "men", "woman": "women", "child": "children",
    "bus": "buses", "bench": "benches", "glass": "glasses", "dish": "dishes",
    "sandwich": "sandwiches", "couch": "couches", "toothbrush": "toothbrushes",
    "mouse": "mice", "sheep": "sheep", "knife": "knives", "broccoli": "broccoli",
    "pizza": "pizzas", "skis": "pairs of skis", "snowboard": "snowboards",
    "scissors": "pairs of scissors",
}


def pluralize_category(name: str) -> str:
    if name in PLURAL_MAP:
        return PLURAL_MAP[name]
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    return name + "s"


def normalize_vqa_answer(ans: str) -> str:
    """Standard VQAv2 answer normalization used consistently across training targets and scoring."""
    if not ans:
        return ""
    ans = ans.replace("\n", " ").replace("\t", " ").strip().lower()
    # Strip periods except decimal points
    ans = re.sub(r"(?<!\d)\.(?!\d)", "", ans)
    # Strip commas between digits
    ans = re.sub(r"(\d),(\d)", r"\1\2", ans)
    # Replace punctuation (except apostrophe and decimal period) with space
    ans = re.sub(r"[^\w\s'\.]", " ", ans)
    words = []
    for w in ans.split():
        w_clean = w.replace("'", "")
        if w_clean in CONTRACTIONS:
            w = CONTRACTIONS[w_clean]
        if w in ARTICLES:
            continue
        w = NUMBER_MAP.get(w, w)
        words.append(w)
    return " ".join(words).strip()


def build_task2_splits(
    coco_train_captions_json: str | Path,
    coco_val_captions_json: str | Path,
    coco_train_instances_json: str | Path,
    coco_val_instances_json: str | Path,
    vqa_train_questions_json: str | Path,
    vqa_train_annotations_json: str | Path,
    vqa_train_comp_pairs_json: str | Path,
    vqa_val_questions_json: str | Path,
    vqa_val_annotations_json: str | Path,
    vqa_val_comp_pairs_json: str | Path,
    output_dir: str | Path,
    seed: int = 6304,
) -> Dict[str, Any]:
    """
    Build all Task 2 datasets and splits using seed 6304 while strictly enforcing
    image-disjointness between train2014, dev (from val2014), and final test (from val2014).
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Select Natural VQAv2 complementary pairs first (since they have strict 8/10 consensus & pair constraints)
    def select_vqa_pairs(q_path, a_path, cp_path, target_counts: Dict[str, int], forbidden_imgs: Set[int]):
        with open(q_path, "r", encoding="utf-8") as f:
            q_data = {q["question_id"]: q for q in json.load(f)["questions"]}
        with open(a_path, "r", encoding="utf-8") as f:
            a_data = {a["question_id"]: a for a in json.load(f)["annotations"]}
        with open(cp_path, "r", encoding="utf-8") as f:
            cp_list = json.load(f)

        by_type = {"yes/no": [], "number": [], "other": []}
        for qid1, qid2 in cp_list:
            if qid1 not in q_data or qid2 not in q_data or qid1 not in a_data or qid2 not in a_data:
                continue
            q1, q2 = q_data[qid1], q_data[qid2]
            a1, a2 = a_data[qid1], a_data[qid2]
            img1, img2 = int(q1["image_id"]), int(q2["image_id"])
            if img1 == img2 or img1 in forbidden_imgs or img2 in forbidden_imgs:
                continue
            ans_type = a1.get("answer_type", "other")
            if ans_type not in by_type:
                continue

            raw1 = [x["answer"] for x in a1["answers"]]
            raw2 = [x["answer"] for x in a2["answers"]]
            norm1 = [normalize_vqa_answer(x) for x in raw1]
            norm2 = [normalize_vqa_answer(x) for x in raw2]
            c1_ans, c1_cnt = Counter(norm1).most_common(1)[0]
            c2_ans, c2_cnt = Counter(norm2).most_common(1)[0]

            if c1_cnt < 8 or c2_cnt < 8 or c1_ans == c2_ans or not c1_ans or not c2_ans:
                continue

            orig_target1 = next(r for r, n in zip(raw1, norm1) if n == c1_ans)
            orig_target2 = next(r for r, n in zip(raw2, norm2) if n == c2_ans)

            by_type[ans_type].append({
                "pair_id": f"vqa_{qid1}_{qid2}",
                "question_id_a": qid1,
                "question_id_b": qid2,
                "image_a_id": img1,
                "image_b_id": img2,
                "question": q1["question"],
                "question_type": ans_type,
                "source_type": "natural_vqa",
                "answer_a": orig_target1,
                "answer_b": orig_target2,
                "gt_answers_a": raw1,
                "gt_answers_b": raw2,
            })

        selected_pairs = []
        used_imgs: Set[int] = set()
        for qt in ("yes/no", "number", "other"):
            pool = by_type[qt]
            perm = rng.permutation(len(pool)).tolist()
            need = target_counts[qt]
            picked = []
            for idx in perm:
                cand = pool[idx]
                if cand["image_a_id"] in forbidden_imgs or cand["image_b_id"] in forbidden_imgs:
                    continue
                picked.append(cand)
                used_imgs.add(cand["image_a_id"])
                used_imgs.add(cand["image_b_id"])
                if len(picked) >= need:
                    break
            selected_pairs.extend(picked)
        return selected_pairs, used_imgs

    train_vqa_pairs, train_vqa_imgs = select_vqa_pairs(
        vqa_train_questions_json, vqa_train_annotations_json, vqa_train_comp_pairs_json,
        {"yes/no": 640, "number": 256, "other": 640}, forbidden_imgs=set()
    )
    dev_vqa_pairs, dev_vqa_imgs = select_vqa_pairs(
        vqa_val_questions_json, vqa_val_annotations_json, vqa_val_comp_pairs_json,
        {"yes/no": 12, "number": 8, "other": 12}, forbidden_imgs=set()
    )
    test_vqa_pairs, test_vqa_imgs = select_vqa_pairs(
        vqa_val_questions_json, vqa_val_annotations_json, vqa_val_comp_pairs_json,
        {"yes/no": 36, "number": 24, "other": 36}, forbidden_imgs=dev_vqa_imgs
    )

    # 2. Build Programmatic COCO Grounding Pairs (Presence and Counting)
    def build_grounding_pools(instances_json_path, forbidden_imgs: Set[int], max_presence: int, max_counting: int):
        with open(instances_json_path, "r", encoding="utf-8") as f:
            inst = json.load(f)
        cats = {c["id"]: c["name"] for c in inst["categories"]}
        all_imgs = sorted(
            [int(im["id"]) for im in inst["images"] if int(im["id"]) not in forbidden_imgs]
        )
        img_cat_counts: Dict[int, Counter] = defaultdict(Counter)
        for ann in inst["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            iid = int(ann["image_id"])
            if iid not in forbidden_imgs:
                img_cat_counts[iid][int(ann["category_id"])] += 1

        all_imgs_set = set(all_imgs)
        presence_pairs, counting_pairs = [], []
        used_imgs: Set[int] = set()

        cat_ids = sorted(cats.keys())
        rng.shuffle(cat_ids)

        # Presence pairs: "Does this image contain [object]?" (count >= 1 vs count == 0)
        for cid in cat_ids:
            cname = cats[cid]
            pos_imgs = [i for i in all_imgs if img_cat_counts[i][cid] >= 1]
            neg_imgs = [i for i in all_imgs if img_cat_counts[i][cid] == 0]
            if not pos_imgs or not neg_imgs:
                continue
            rng.shuffle(pos_imgs)
            rng.shuffle(neg_imgs)
            n_take = min(len(pos_imgs), len(neg_imgs), max(2, max_presence // max(1, len(cat_ids)) + 2))
            for k in range(n_take):
                ia, ib = pos_imgs[k], neg_imgs[k]
                presence_pairs.append({
                    "pair_id": f"pres_{cid}_{ia}_{ib}",
                    "image_a_id": ia,
                    "image_b_id": ib,
                    "question": f"Does this image contain {cname}?",
                    "question_type": "presence",
                    "source_type": "presence",
                    "answer_a": "yes",
                    "answer_b": "no",
                })
                used_imgs.update((ia, ib))
                if len(presence_pairs) >= max_presence:
                    break
            if len(presence_pairs) >= max_presence:
                break

        # Counting pairs: "How many [objects] can you see?" (different counts in 1..4)
        for cid in cat_ids:
            cname_plural = pluralize_category(cats[cid])
            by_cnt = {c: [i for i in all_imgs if img_cat_counts[i][cid] == c] for c in (1, 2, 3, 4)}
            valid_cnts = [c for c in (1, 2, 3, 4) if len(by_cnt[c]) > 0]
            if len(valid_cnts) < 2:
                continue
            for c1 in valid_cnts:
                for c2 in valid_cnts:
                    if c1 >= c2:
                        continue
                    l1, l2 = list(by_cnt[c1]), list(by_cnt[c2])
                    rng.shuffle(l1)
                    rng.shuffle(l2)
                    for ia, ib in zip(l1[:15], l2[:15]):
                        counting_pairs.append({
                            "pair_id": f"cnt_{cid}_{ia}_{ib}",
                            "image_a_id": ia,
                            "image_b_id": ib,
                            "question": f"How many {cname_plural} can you see?",
                            "question_type": "counting",
                            "source_type": "counting",
                            "answer_a": str(c1),
                            "answer_b": str(c2),
                        })
                        used_imgs.update((ia, ib))
                        if len(counting_pairs) >= max_counting:
                            break
                    if len(counting_pairs) >= max_counting:
                        break
                if len(counting_pairs) >= max_counting:
                    break
            if len(counting_pairs) >= max_counting:
                break

        return presence_pairs[:max_presence], counting_pairs[:max_counting], used_imgs

    train_pres, train_cnt, train_gnd_imgs = build_grounding_pools(
        coco_train_instances_json, forbidden_imgs=set(), max_presence=2000, max_counting=1000
    )
    dev_pres, dev_cnt, dev_gnd_imgs = build_grounding_pools(
        coco_val_instances_json, forbidden_imgs=test_vqa_imgs, max_presence=12, max_counting=12
    )
    test_pres, test_cnt, test_gnd_imgs = build_grounding_pools(
        coco_val_instances_json,
        forbidden_imgs=(dev_vqa_imgs | dev_gnd_imgs),
        max_presence=32,
        max_counting=32,
    )

    # 3. Select COCO Captions (6,000 train images x 3 captions = 18,000; 96 dev images x 1 caption = 96)
    def select_coco_captions(cap_json_path, num_images: int, caps_per_img: int, forbidden_imgs: Set[int]):
        with open(cap_json_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        by_img: Dict[int, List[str]] = defaultdict(list)
        for ann in c_data["annotations"]:
            iid = int(ann["image_id"])
            if iid not in forbidden_imgs and ann.get("caption", "").strip():
                by_img[iid].append(ann["caption"].strip())
        eligible_imgs = sorted([iid for iid, caps in by_img.items() if len(caps) >= caps_per_img])
        chosen_imgs = rng.choice(eligible_imgs, size=min(num_images, len(eligible_imgs)), replace=False).tolist()
        examples = []
        for iid in chosen_imgs:
            for c_text in by_img[iid][:caps_per_img]:
                examples.append({
                    "source_type": "caption",
                    "image_id": int(iid),
                    "question": "Describe this image.",
                    "target_text": c_text,
                    "is_caption": True,
                })
        return examples, set(chosen_imgs)

    train_captions, train_cap_imgs = select_coco_captions(
        coco_train_captions_json, num_images=6000, caps_per_img=3, forbidden_imgs=set()
    )
    dev_captions, dev_cap_imgs = select_coco_captions(
        coco_val_captions_json,
        num_images=96,
        caps_per_img=1,
        forbidden_imgs=(test_vqa_imgs | test_gnd_imgs),
    )

    # Strictly verify image-disjointness between train, dev, and final test
    all_train_imgs = train_vqa_imgs | train_gnd_imgs | train_cap_imgs
    all_dev_imgs = dev_vqa_imgs | dev_gnd_imgs | dev_cap_imgs
    all_test_imgs = test_vqa_imgs | test_gnd_imgs

    assert all_train_imgs.isdisjoint(all_dev_imgs), "Train and Dev image sets overlap!"
    assert all_train_imgs.isdisjoint(all_test_imgs), "Train and Test image sets overlap!"
    assert all_dev_imgs.isdisjoint(all_test_imgs), "Dev and Test image sets overlap!"

    splits_payload = {
        "seed": seed,
        "train_pools": {
            "captions": train_captions,
            "presence_pairs": train_pres,
            "counting_pairs": train_cnt,
            "natural_vqa_pairs": train_vqa_pairs,
        },
        "dev_suites": {
            "captions": dev_captions,
            "grounding_pairs": dev_pres + dev_cnt,
            "natural_vqa_pairs": dev_vqa_pairs,
        },
        "test_suites": {
            "grounding_test": test_pres + test_cnt,
            "natural_vqa_test": test_vqa_pairs,
        },
        "image_ids": {
            "train2014": sorted(list(all_train_imgs)),
            "val2014_dev": sorted(list(all_dev_imgs)),
            "val2014_test": sorted(list(all_test_imgs)),
        },
    }
    save_json(splits_payload, out_dir / "task2_splits_and_ids.json")
    return splits_payload
