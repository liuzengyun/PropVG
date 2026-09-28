#!/usr/bin/env python3
"""Create a source-stratified, scene-grouped AerialSense split.

This is a new experimental protocol, not the unpublished GrabVG sample list.
Existing annotations and teacher files are never overwritten. Teacher scores
and predictions are copied unchanged; only sample_id follows the new order.
"""

import argparse
import collections
import copy
import hashlib
import json
import math
import random
import re
import shutil
from pathlib import Path

from convert_aerialsense import load_record, natural_key


SOURCES = ("CODrone", "VisDrone", "AerialVG", "refGeo", "University-1652")
OLD_FILES = (
    "annotation/vg_train_odvg.jsonl", "annotation/vg_val_odvg.jsonl",
    "annotation/vg_test_odvg.jsonl", "teacher_aerialsense_train.json",
    "teacher_aerialsense_val.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_key(record):
    return record["filename"], record["grounding"]["caption"].strip()


def scene_key(source, original_name):
    """Conservative capture/identity groups inferred from source filenames."""
    stem = Path(original_name).stem
    if source == "CODrone":
        match = re.fullmatch(r"(.+)_frame_\d+", stem)
        if not match:
            raise ValueError(f"Unknown CODrone video filename: {original_name}")
        return "CODrone:video:" + re.sub(r"\s+", "", match.group(1)).lower()
    if source in ("VisDrone", "AerialVG"):
        if source == "AerialVG":
            stem = re.sub(r"^\d+_(?=\d{7}_\d+_d_\d+$)", "", stem)
        match = re.fullmatch(r"(\d{7})_\d+_d_\d+", stem)
        if not match:
            raise ValueError(f"Unknown {source} sequence filename: {original_name}")
        # AerialVG images originate from VisDrone: share the same namespace.
        return "VisDrone:sequence:" + match.group(1)
    if source == "University-1652":
        match = re.fullmatch(r"(\d+)_image-\d+", stem)
        if not match:
            raise ValueError(f"Unknown University building filename: {original_name}")
        return "University-1652:building:" + match.group(1)
    if source == "refGeo":
        # These files have no reliable scene/video ID. Do not invent one from
        # unrelated DJI frame numbers or claim geographic independence.
        return "refGeo:image:" + Path(original_name).name.lower()
    raise ValueError(f"Unknown source: {source}")


class UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def build_units(entries):
    images = {entry["record"]["filename"] for entry in entries}
    union = UnionFind(images)
    representatives = {}
    for entry in entries:
        filename = entry["record"]["filename"]
        # Exact duplicate image files also stay together, including aliases
        # appearing in different source datasets or under different names.
        for key in (("scene", entry["scene_id"]), ("image_sha256", entry["image_sha256"])):
            if key in representatives:
                union.union(filename, representatives[key])
            else:
                representatives[key] = filename
    grouped = collections.defaultdict(list)
    for index, entry in enumerate(entries):
        grouped[union.find(entry["record"]["filename"])].append(index)
    units = []
    for root, indices in sorted(grouped.items()):
        counts = collections.Counter(entries[index]["source"] for index in indices)
        units.append({
            "id": root, "indices": indices,
            "counts": tuple(counts[source] for source in SOURCES),
            "scenes": sorted({entries[index]["scene_id"] for index in indices}),
            "images": sorted({entries[index]["record"]["filename"] for index in indices}),
        })
    return units


def balanced_assignment(units, ratio, seed, trials=128):
    """Balance annotation counts without splitting any scene/duplicate group.

    Randomized local search uses counts only, never labels, teacher quality,
    model predictions or validation accuracy. Sorting + a local RNG make it
    reproducible independently of filesystem order and global random state.
    """
    totals = tuple(sum(unit["counts"][j] for unit in units) for j in range(len(SOURCES)))
    if any(total < 2 for total in totals):
        raise ValueError("Each source must have at least two annotations")
    for j, source in enumerate(SOURCES):
        if sum(unit["counts"][j] > 0 for unit in units) < 2:
            raise ValueError(f"{source} has fewer than two independent scene groups")

    def cost(counts):
        result = sum((count / total - ratio) ** 2 for count, total in zip(counts, totals))
        result += (sum(counts) / sum(totals) - ratio) ** 2
        # Both train and held-out must contain every source.
        result += sum(count <= 0 or count >= total for count, total in zip(counts, totals))
        return result

    rng = random.Random(seed)
    best = None
    for _ in range(trials):
        assigned = [rng.random() < ratio for _ in units]
        counts = tuple(sum(unit["counts"][j] for unit, flag in zip(units, assigned) if flag)
                       for j in range(len(SOURCES)))
        value = cost(counts)
        for _ in range(12):
            improved = False
            order = list(range(len(units)))
            rng.shuffle(order)
            for index in order:
                sign = -1 if assigned[index] else 1
                candidate = tuple(a + sign * b for a, b in zip(counts, units[index]["counts"]))
                candidate_cost = cost(candidate)
                if candidate_cost < value - 1e-15:
                    assigned[index] = not assigned[index]
                    counts, value, improved = candidate, candidate_cost, True
            if not improved:
                break
        if best is None or value < best[0] - 1e-15:
            best = value, assigned[:], counts

    value, assigned, counts = best
    # A swap can improve balance when no individual group toggle can do so.
    for _ in range(4):
        best_swap = None
        for left, left_unit in enumerate(units):
            if not assigned[left]:
                continue
            for right, right_unit in enumerate(units):
                if assigned[right]:
                    continue
                candidate = tuple(a - b + c for a, b, c in
                                  zip(counts, left_unit["counts"], right_unit["counts"]))
                candidate_cost = cost(candidate)
                if candidate_cost < value - 1e-15 and (best_swap is None or candidate_cost < best_swap[0] - 1e-15):
                    best_swap = candidate_cost, left, right, candidate
        if best_swap is None:
            break
        value, left, right, counts = best_swap
        assigned[left], assigned[right] = False, True
    if any(count <= 0 or count >= total for count, total in zip(counts, totals)):
        raise ValueError("Cannot put every source in both splits without breaking scene groups")
    return assigned


def read_inputs(root):
    image_sizes, image_hashes = {}, {}
    entries = []
    paths = sorted((root / "question").glob("visual_grounding_*.json"), key=natural_key)
    if not paths:
        raise ValueError("No original visual-grounding questions found")
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        record, _, _ = load_record(path, root / "images", image_sizes)
        dataset = source["dataset"]
        filename = record["filename"]
        if filename not in image_hashes:
            image_hashes[filename] = sha256_file(root / "images" / filename)
        entries.append({
            "record": record, "source": dataset,
            "scene_id": scene_key(dataset, source["old_image_name"]),
            "source_question": path.name, "original_image_name": source["old_image_name"],
            "image_sha256": image_hashes[filename],
        })
    keys = [record_key(entry["record"]) for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate image/expression keys in original questions")

    original_records, teachers = {}, {}
    for split in ("train", "val"):
        records = [json.loads(line) for line in
                   (root / "annotation" / f"vg_{split}_odvg.jsonl").read_text(encoding="utf-8").splitlines()]
        predictions = json.loads((root / f"teacher_aerialsense_{split}.json").read_text(encoding="utf-8"))
        if len(records) != len(predictions):
            raise ValueError(f"{split}: annotation/teacher count mismatch")
        for sample_id, (record, teacher) in enumerate(zip(records, predictions)):
            key = record_key(record)
            if key in teachers:
                raise ValueError(f"Duplicate teacher key: {key}")
            if (teacher["sample_id"] != sample_id or teacher["filename"] != key[0]
                    or teacher["caption"].strip() != key[1]
                    or teacher["width"] != record["width"] or teacher["height"] != record["height"]):
                raise ValueError(f"Teacher alignment mismatch at {split}:{sample_id}")
            original_records[key] = record
            teachers[key] = (teacher, split, sample_id)
    old_test = [json.loads(line) for line in
                (root / "annotation/vg_test_odvg.jsonl").read_text(encoding="utf-8").splitlines()]
    old_val = [record for key, record in original_records.items() if teachers[key][1] == "val"]
    if old_test != old_val:
        raise ValueError("Expected the previous val and test files to be identical")
    if set(keys) != set(teachers):
        raise ValueError("Original question coverage differs from combined old annotation/teacher coverage")
    for entry in entries:
        key = record_key(entry["record"])
        if entry["record"] != original_records[key]:
            raise ValueError(f"Conversion differs from previous annotation: {key}")
        teacher, previous_split, previous_id = teachers[key]
        if not teacher.get("predictions"):
            raise ValueError(f"No teacher predictions for {key}")
        for prediction in teacher["predictions"]:
            box = prediction.get("bbox_xyxy")
            if (not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in box)
                    or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError(f"Invalid teacher box for {key}")
        entry.update(teacher=teacher, previous_split=previous_split, previous_sample_id=previous_id)
    return entries


def make_payload(entries, units, assigned, ratio, seed):
    membership = {}
    for unit, train in zip(units, assigned):
        for index in unit["indices"]:
            membership[index] = ("train" if train else "val", unit["id"])
    records = {"train": [], "val": []}
    teachers = {"train": [], "val": []}
    samples = []
    for index, entry in enumerate(entries):
        split, group = membership[index]
        sample_id = len(records[split])
        records[split].append(entry["record"])
        teacher = copy.deepcopy(entry["teacher"])
        teacher["sample_id"] = sample_id
        teachers[split].append(teacher)
        samples.append({
            "split": split, "sample_id": sample_id, "group_id": group,
            **{key: entry[key] for key in ("source", "scene_id", "source_question",
                                           "original_image_name", "image_sha256",
                                           "previous_split", "previous_sample_id")},
            "filename": entry["record"]["filename"],
        })
    for key in ("filename", "scene_id", "image_sha256", "group_id"):
        sets = [{sample[key] for sample in samples if sample["split"] == split} for split in ("train", "val")]
        if sets[0] & sets[1]:
            raise AssertionError(f"Cross-split leakage: {key}")
    statistics = {}
    for source in SOURCES:
        statistics[source] = {}
        for split in ("train", "val"):
            subset = [sample for sample in samples if sample["source"] == source and sample["split"] == split]
            statistics[source][split] = {
                "annotations": len(subset), "images": len({s["filename"] for s in subset}),
                "scene_groups": len({s["group_id"] for s in subset}),
            }
            if not subset:
                raise AssertionError(f"Empty source/split: {source}/{split}")
        train_count = statistics[source]["train"]["annotations"]
        total_count = train_count + statistics[source]["val"]["annotations"]
        statistics[source]["actual_train_ratio"] = train_count / total_count
    manifest = {
        "protocol": "source_stratified_scene_grouped", "seed": seed,
        "splitter_version": 1,
        "requested_annotation_train_ratio": ratio,
        "actual_annotation_train_ratio": len(records["train"]) / len(entries),
        "val_equals_test": True, "teacher_predictions_changed": False,
        "counts": {
            split: {
                "annotations": len(records[split]),
                "images": len({s["filename"] for s in samples if s["split"] == split}),
                "groups": len({s["group_id"] for s in samples if s["split"] == split}),
            } for split in ("train", "val")
        },
        "previous_train_in_new_holdout": sum(s["previous_split"] == "train" and s["split"] == "val" for s in samples),
        "previous_holdout_in_new_train": sum(s["previous_split"] == "val" and s["split"] == "train" for s in samples),
        "grouping": {
            "CODrone": "same filename-inferred capture video",
            "VisDrone/AerialVG": "same shared original VisDrone sequence",
            "University-1652": "same building ID across image viewpoints",
            "refGeo": "same original image; geographic scene ID unavailable",
            "duplicates": "same SHA256 image bytes; no near-duplicate claim",
        },
        "statistics": statistics, "samples": samples,
    }
    return records, teachers, manifest


def serialize_payload(records, teachers):
    return {
        "annotation/vg_train_odvg.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records["train"]),
        "annotation/vg_val_odvg.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records["val"]),
        "annotation/vg_test_odvg.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records["val"]),
        "teacher_train.json": json.dumps(teachers["train"], ensure_ascii=False, indent=2) + "\n",
        "teacher_val.json": json.dumps(teachers["val"], ensure_ascii=False, indent=2) + "\n",
    }


def print_summary(manifest):
    print(f"Protocol: {manifest['protocol']}; seed={manifest['seed']}")
    for source, values in manifest["statistics"].items():
        print(f"{source}: train={values['train']['annotations']} ({values['train']['images']} images), "
              f"val/test={values['val']['annotations']} ({values['val']['images']} images), "
              f"train ratio={values['actual_train_ratio']:.2%}")
    print(f"Overall annotation train ratio: {manifest['actual_annotation_train_ratio']:.2%}")
    print(f"Totals: {manifest['counts']}")
    print("No cross-split image, inferred scene, or exact image-byte duplicates; val=test.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/AerialSense"))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New directory; default DATA_ROOT/splits/scene_seedSEED")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--dry-run", action="store_true", help="Validate and calculate without writing files")
    parser.add_argument("--verify-only", action="store_true", help="Check existing output against deterministic recomputation")
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be between 0 and 1")
    if args.dry_run and args.verify_only:
        parser.error("--dry-run and --verify-only are mutually exclusive")
    root = args.data_root.resolve()
    output = (args.output_dir or root / "splits" / f"scene_seed{args.seed}").resolve()
    # New output only, so neither old data nor a previous split can be erased.
    if not args.verify_only and not args.dry_run and output.exists():
        parser.error("Output already exists; use --verify-only or choose another --output-dir")
    entries = read_inputs(root)
    units = build_units(entries)
    print(f"Read {len(entries)} annotations; {len(units)} indivisible groups", flush=True)
    assigned = balanced_assignment(units, args.train_ratio, args.seed)
    records, teachers, manifest = make_payload(entries, units, assigned, args.train_ratio, args.seed)
    payload = serialize_payload(records, teachers)
    manifest["original_files_sha256"] = {name: sha256_file(root / name) for name in OLD_FILES}
    manifest["output_files_sha256"] = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in payload.items()}
    if args.verify_only:
        saved = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
        if saved != manifest:
            raise AssertionError("Saved manifest differs from deterministic recomputation")
        for name, digest in manifest["output_files_sha256"].items():
            if sha256_file(output / name) != digest:
                raise AssertionError(f"Output contents changed: {name}")
        for name, digest in manifest["original_files_sha256"].items():
            if sha256_file(output / "previous_ordered_split" / name) != digest:
                raise AssertionError(f"Backup differs from original: {name}")
        print("Verification passed: reproducibility, output hashes, original backups, and disjoint groups.")
    elif not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)
        # Back up only the five explicitly identified old annotation files.
        # Images, source questions, checkpoints, and running jobs are untouched.
        for name in OLD_FILES:
            destination = output / "previous_ordered_split" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        for name, text in payload.items():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"New split: {output}")
    print_summary(manifest)


if __name__ == "__main__":
    main()
