"""Frozen-baseline CRV training, with optional offline cache. Legacy train.py is untouched."""
import argparse
import copy
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from propvg.apis.crv_cached import (
    CACHE_VERSION, CachedCRVDataset, assemble_checkpoint, atomic_json_save, atomic_torch_save,
    build_crv, cache_identity, cached_predictions, call_cached_crv, collate_cached,
    cleanup_feature_caches, extract_frozen_inputs, feature_cache_lock, file_sha256, live_feature_batch,
    load_configured_baseline, load_frozen_baseline, make_record, validate_cache_identity,
)
from propvg.apis.test import aerialvg_box_metrics
from propvg.apis.train import set_random_seed
from propvg.datasets import build_dataset
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
from propvg.models import build_model
from propvg.utils import get_root_logger
from propvg.utils.experiment_tracking import (
    finish_tracking, initialize_tracking, log_metrics, log_summary, set_tracking_context,
)


def move_inputs(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_inputs(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_inputs(item, device) for item in value]
    return value


def unpack_inputs(containers, device):
    return {key: value.data[0] if value.cpu_only else move_inputs(value.data[0], device)
            for key, value in containers.items()}


def validate_config(cfg):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("First offline CRV version is single-GPU; do not use dist_train.sh/torchrun")
    if not cfg.get("offline_crv") or not cfg.model.head.get("enable_crv", False):
        raise ValueError("Use the standalone PropVG-crv-cached.py config with enable_crv=True")
    if cfg.model.post_params.get("with_nms", False):
        raise ValueError("First cached evaluator requires with_nms=False, as in the AerialVG baseline")
    for key in ("use_cache", "cleanup_cache_after_training"):
        if key in cfg.offline_crv and not isinstance(cfg.offline_crv[key], bool):
            raise ValueError("offline_crv.{} must be True or False".format(key))
    if cfg.offline_crv.get("use_cache", True) and cfg.offline_crv.cache_dtype not in ("float32", "float16"):
        raise ValueError("cache_dtype must be float32 or float16")
    if cfg.offline_crv.max_epochs < 1 or cfg.offline_crv.batch_size < 1:
        raise ValueError("max_epochs and batch_size must be positive")
    allowed = {"LoadAerialVGAnnotations", "Resize", "Pad", "Normalize", "DefaultFormatBundle", "CollectData"}
    for split in ("train", "val", "test"):
        if split not in cfg.data:
            continue
        dataset = cfg.data[split]
        if dataset.get("teacher_annsfile"):
            raise ValueError("Offline CRV requires only original refer GT; disable teacher loading")
        for operation in dataset.pipeline:
            if operation["type"] not in allowed:
                raise ValueError("Offline cache requires fixed deterministic preprocessing: {}".format(operation["type"]))
            if operation["type"] == "LoadAerialVGAnnotations" and (
                    operation.get("use_teacher_boxes") or operation.get("use_auxiliary_boxes")):
                raise ValueError("Offline CRV uses only original refer GT, not auxiliary/teacher targets")
    if Path(cfg.data.train.annsfile).resolve() in {
            Path(cfg.data[split].annsfile).resolve() for split in ("val", "test") if split in cfg.data}:
        raise ValueError("CRV training annotations must be separate from validation/test")


def extract_caches(cfg, splits, checkpoint_hash, device, logger):
    model = None
    for split in splits:
        fingerprint, details = cache_identity(cfg, split, checkpoint_hash)
        directory = Path(cfg.offline_crv.cache_dir) / split
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            try:
                accepted = validate_cache_identity(manifest, fingerprint, details)
            except ValueError as error:
                raise ValueError("Existing {} cache belongs to different weights/config/data or incompatible feature code: {} "
                                 "Old cache is not overwritten.".format(split, error)) from error
            if accepted != fingerprint:
                logger.info("Compatible legacy %s cache: feature identity verified; reuse without rewriting samples", split)
            fingerprint = accepted  # Resume old caches using the SAME per-record identity.
        elif any(directory.iterdir()):
            raise ValueError("Cache directory is nonempty but has no manifest: {}".format(directory))
        else:
            manifest = dict(version=CACHE_VERSION, fingerprint=fingerprint, details=details, complete=False)
        dataset = build_dataset(cfg.data[split])
        count = len(dataset)
        if not count:
            raise ValueError("Empty {} dataset".format(split))
        if manifest.get("count", count) != count:
            raise ValueError("Cached dataset length changed")
        missing = [index for index in range(count) if not (directory / ("{:08d}.pt".format(index))).is_file()]
        if not missing and manifest.get("complete"):
            logger.info("Reuse complete %s cache: %d samples (no baseline forward)", split, count)
            continue
        manifest.update(count=count, complete=False)
        atomic_json_save(manifest, manifest_path)
        logger.info("Cache %s: %d samples, %d remaining; dtype=%s", split, count, len(missing), cfg.offline_crv.cache_dtype)
        if missing and model is None:
            model = build_frozen_model(cfg, dataset, load_configured_baseline(cfg), device)
        batch_size = int(cfg.offline_crv.extract_batch_size)
        loader = DataLoader(Subset(dataset, missing), batch_size=batch_size, shuffle=False,
                            num_workers=int(cfg.offline_crv.extract_workers),
                            collate_fn=partial(collate, samples_per_gpu=batch_size))
        offset = 0
        for containers in tqdm(loader, desc="Cache " + split, dynamic_ncols=True):
            inputs = unpack_inputs(containers, device)
            features = extract_frozen_inputs(model, inputs)
            for index in range(len(inputs["img_metas"])):
                sample_index = missing[offset]
                record = make_record(features, inputs, index, sample_index, fingerprint, cfg.offline_crv.cache_dtype)
                if offset == 0:
                    sample_bytes = sum(value.numel() * value.element_size() for value in record.values()
                                       if isinstance(value, torch.Tensor))
                    estimated = int(sample_bytes * len(missing) * 1.10)
                    free = shutil.disk_usage(directory).free
                    logger.info("%s estimated remaining cache %.1f GiB; free disk %.1f GiB",
                                split, estimated / 2**30, free / 2**30)
                    if estimated > free:
                        raise OSError("Insufficient cache disk space; use another cache_dir or cache_dtype=float16")
                atomic_torch_save(record, directory / ("{:08d}.pt".format(sample_index)))
                offset += 1
            del features, inputs
        manifest["complete"] = True
        atomic_json_save(manifest, manifest_path)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cache_loader(cfg, split, checkpoint_hash, shuffle=False, generator=None):
    fingerprint, details = cache_identity(cfg, split, checkpoint_hash)
    dataset = CachedCRVDataset(Path(cfg.offline_crv.cache_dir) / split, fingerprint, details)
    if dataset.legacy_compatible:
        get_root_logger().info("Reuse compatible legacy %s features; no baseline forward or cache rewrite", split)
    return DataLoader(dataset, batch_size=int(cfg.offline_crv.batch_size), shuffle=shuffle,
                      num_workers=int(cfg.offline_crv.workers), collate_fn=collate_cached,
                      pin_memory=False, generator=generator)


def build_frozen_model(cfg, dataset, state, device):
    options = copy.deepcopy(dict(cfg.model))
    options["head"]["enable_crv"] = False
    options["vis_enc"]["pretrain"] = None
    model = build_model(options, word_emb=dataset.word_emb, num_token=dataset.num_token)
    return load_frozen_baseline(model, state).to(device)


class OnlineFeatureLoader:
    """One frozen grounding forward per batch, on each pass over the dataset."""
    def __init__(self, loader, model, device):
        self.loader, self.model, self.device = loader, model, device
        self.dataset = loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.model.eval()  # CRV.train() never propagates to this separate model.
        for containers in self.loader:
            inputs = unpack_inputs(containers, self.device)
            features = extract_frozen_inputs(self.model, inputs)
            batch = live_feature_batch(features, inputs)
            del features, inputs
            yield batch
            del batch


def online_loader(cfg, split, model, device, shuffle=False, generator=None, dataset=None):
    dataset = build_dataset(cfg.data[split]) if dataset is None else dataset
    batch_size = int(cfg.offline_crv.batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=int(cfg.offline_crv.workers),
                        collate_fn=partial(collate, samples_per_gpu=batch_size), generator=generator)
    return OnlineFeatureLoader(loader, model, device)


@torch.no_grad()
def evaluate_cached(crv, loader, device, prefix, logger):
    crv.eval()
    sums = {name: torch.zeros(3, dtype=torch.float64) for name in ("crs", "crv", "fusion")}
    corrected, harmed, count, cf_count = 0, 0, 0, 0
    for batch in tqdm(loader, desc="Evaluate " + prefix, dynamic_ncols=True):
        probabilities, priority, _, info = call_cached_crv(crv, batch, device)
        before = cached_predictions(batch, batch["probabilities"])
        after = cached_predictions(batch, probabilities, priority)
        cf_order = batch["probabilities"].detach().cpu().squeeze(-1).argsort(dim=1, descending=True, stable=True)
        for index, details in enumerate(info["crv_samples"]):
            if not details.get("skipped", False):
                indices = details["indices"].cpu()
                cf_order[index, :len(indices)] = indices[
                    details["drops"].detach().cpu().argsort(descending=True, stable=True)]
        cf_priority = torch.empty_like(cf_order, dtype=torch.float32)
        cf_priority.scatter_(1, cf_order, torch.arange(cf_order.shape[1], 0, -1)[None].to(cf_priority).expand_as(cf_priority))
        independent = cached_predictions(batch, batch["probabilities"], cf_priority[..., None])
        for index, (base, fused, target) in enumerate(zip(before, after, batch["refer_gt_pixel"])):
            first = torch.tensor(aerialvg_box_metrics([base], [target]), dtype=torch.float64)
            final = torch.tensor(aerialvg_box_metrics([fused], [target]), dtype=torch.float64)
            sums["crs"] += first
            sums["fusion"] += final
            corrected += int(first[0] == 0 and final[0] > 0)
            harmed += int(first[0] > 0 and final[0] == 0)
            details = info["crv_samples"][index]
            if not details.get("skipped", False):
                # Independent CRV: same Top-K candidate set, no learned alpha.
                sums["crv"] += torch.tensor(aerialvg_box_metrics([independent[index]], [target]), dtype=torch.float64)
                cf_count += 1
            else:
                sums["crv"] += first  # Routing skip: unchanged CRS, also for independent-CRV mode.
            count += 1
    if count == 0:
        raise ValueError("Empty evaluation cache")
    result = {name: (total / count).tolist() for name, total in sums.items()}
    result.update(corrected=corrected, harmed=harmed, net=corrected - harmed, samples=count, cf_evaluated=cf_count)
    for mode in ("crs", "crv", "fusion"):
        top1, top5, miou = result[mode]
        logger.info("%s %s: Top-1 Acc@0.5 %.2f, Top-5 Acc@0.5 %.2f, box mIoU %.2f",
                    prefix, mode, top1, top5, miou)
        log_metrics({"{}/{}/{}".format(prefix, mode, key): value for key, value in
                     zip(("top1_acc", "top5_acc", "box_miou"), result[mode])})
    logger.info("%s: corrected=%d harmed=%d net=%+d / %d; CRV-routed=%d", prefix, corrected, harmed,
                corrected - harmed, count, cf_count)
    log_metrics({prefix + "/" + key: result[key] for key in ("corrected", "harmed", "net", "cf_evaluated")})
    return result


def train_cached(cfg, checkpoint_hash, device):
    cfg.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = Path(cfg.work_dir) / cfg.timestamp
    directory.mkdir(parents=True, exist_ok=False)
    cfg.work_dir = str(directory)
    logger = get_root_logger(str(directory / "train_log.txt"))
    cfg.dump(str(directory / "config.py"))
    if cfg.offline_crv.get("backup_code", True):
        shutil.copytree("propvg", directory / "propvg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(__file__, directory / "train_crv_cached.py")
    initialize_tracking(cfg, logger)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    use_cache = cfg.offline_crv.get("use_cache", True)
    baseline, bcem_validation = load_configured_baseline(cfg, return_validation=True)
    atomic_json_save(bcem_validation, directory / "baseline_bcem_validation.json")
    if bcem_validation["contract"]["enabled"]:
        logger.info("Frozen BCEM configuration verified against %s; all BCEM weights/buffers remain frozen",
                    bcem_validation["sources"])
    if cfg.model.head.get("enable_correctness", False):
        logger.info("Preserve pretrained correctness head as frozen baseline state; it is not a CRV routing gate")
    grounding = None
    if use_cache:
        # Never instantiate/move the grounding model to GPU in cached training.
        train = cache_loader(cfg, "train", checkpoint_hash, shuffle=True, generator=generator)
        validation = cache_loader(cfg, "val", checkpoint_hash)
        logger.info("CRV feature source: offline cache; grounding has NO per-epoch forward")
    else:
        dataset = build_dataset(cfg.data.train)
        grounding = build_frozen_model(cfg, dataset, baseline, device)
        train = online_loader(cfg, "train", grounding, device, shuffle=True, generator=generator, dataset=dataset)
        validation = online_loader(cfg, "val", grounding, device)
        logger.info("CRV feature source: online frozen grounding forward per batch; no feature cache is read/written")
    crv = build_crv(cfg).to(device)
    optimizer = torch.optim.Adam(crv.parameters(), lr=float(cfg.offline_crv.lr),
                                 betas=(0.9, 0.98), weight_decay=0)
    best_metrics, best_top1 = None, -1.
    start, steps, total_updates = time.time(), 0, 0
    log_summary({"training_mode": "cached_crv_only" if use_cache else "online_frozen_crv_only",
                 "baseline_sha256": checkpoint_hash,
                 "model/trainable_parameters": sum(p.numel() for p in crv.parameters()),
                 "data/train_samples": len(train.dataset), "data/val_samples": len(validation.dataset)})
    for epoch in range(int(cfg.offline_crv.max_epochs)):
        crv.train()
        history, updates = {}, 0
        started = time.time()
        for batch_index, batch in enumerate(tqdm(train, desc="Train CRV epoch {}".format(epoch + 1), dynamic_ncols=True)):
            optimizer.zero_grad(set_to_none=True)
            _, _, losses, info = call_cached_crv(crv, batch, device, training=True, epoch=epoch)
            loss = sum(losses.values())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite CRV loss; checkpoint not updated")
            if loss.requires_grad:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(crv.parameters(), float(cfg.offline_crv.grad_clip),
                                                      error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                total_updates += 1
            else:
                norm = loss.new_zeros(())  # e.g. every sample was entropy-routed away.
            values = {key: float(value.detach()) for key, value in losses.items()}
            values.update({"crv_" + key: float(info["crv_" + key]) for key in crv.metric_names})
            values["grad_norm"] = float(norm)
            for key, value in values.items():
                history.setdefault(key, []).append(value)
            steps += 1
            if (batch_index + 1) % int(cfg.log_interval) == 0 or batch_index + 1 == len(train):
                means = {key: sum(series) / len(series) for key, series in history.items()}
                logger.info("CRV epoch [%d/%d] [%d/%d] %s", epoch + 1, cfg.offline_crv.max_epochs,
                            batch_index + 1, len(train), ", ".join("{}={:.4f}".format(key, value) for key, value in means.items()))
                set_tracking_context(epoch=epoch + 1, global_step=steps)
                log_metrics({"train/" + key: value for key, value in means.items()})
        result = evaluate_cached(crv, validation, device, "val", logger)
        selected = result["fusion"]
        improved = selected[0] > best_top1
        if improved:
            best_top1, best_metrics = selected[0], selected
        metadata = dict(baseline_sha256=checkpoint_hash,
                        frozen_bcem_config=bcem_validation["contract"],
                        cache_fingerprint=train.dataset.manifest["fingerprint"] if use_cache else None,
                        use_feature_cache=use_cache,
                        top1_acc=selected[0], top5_acc=selected[1], mIoU=selected[2],
                        best_top1_acc=best_top1, running_best_top1_acc=best_top1,
                        best_top5_acc_at_top1=best_metrics[1], best_mIoU_at_top1=best_metrics[2],
                        lr=float(cfg.offline_crv.lr), crv_updates=updates, crv_total_updates=total_updates,
                        config=str(directory / "config.py"))
        checkpoint = assemble_checkpoint(baseline, crv, epoch, optimizer, metadata)
        atomic_torch_save(checkpoint, directory / "latest.pth")
        if improved:
            shutil.copyfile(directory / "latest.pth", directory / "top1_best.pth.tmp")
            os.replace(str(directory / "top1_best.pth.tmp"), str(directory / "top1_best.pth"))
        atomic_json_save(result, directory / "val_epoch_{:03d}.json".format(epoch + 1))
        logger.info("Epoch %d completed in %.1fs, optimizer updates=%d; full checkpoint saved",
                    epoch + 1, time.time() - started, updates)
    if total_updates == 0:
        logger.warning("No CRV optimizer updates occurred: check start_epoch, entropy routing and GT availability")
    test_metrics = None
    if cfg.offline_crv.get("test_after_training", True):
        best = torch.load(directory / "top1_best.pth", map_location="cpu")
        crv.load_state_dict({key[len("head.crv."):]: value for key, value in best["state_dict"].items()
                             if key.startswith("head.crv.")}, strict=True)
        del best
        test = (cache_loader(cfg, "test", checkpoint_hash) if use_cache
                else online_loader(cfg, "test", grounding, device))
        test_result = evaluate_cached(crv, test, device, "test_best", logger)
        test_metrics = test_result["fusion"]
        atomic_json_save(test_result, directory / "test_best.json")
    log_summary({"best_val": best_metrics, "test_best": test_metrics, "seconds": time.time() - start})
    finish_tracking()
    if cfg.offline_crv.get("rename_with_metrics", True):
        suffix = "+val_top1={:.2f}+val_top5={:.2f}+val_miou={:.2f}".format(*best_metrics)
        if test_metrics is not None:
            suffix += "+test_top1={:.2f}+test_top5={:.2f}+test_miou={:.2f}".format(*test_metrics)
        destination = directory.with_name(directory.name + suffix)
        directory.rename(destination)
        directory = destination
    # No finally cleanup: retain cache on crashes/interruption/test/save errors.
    if use_cache and cfg.offline_crv.get("cleanup_cache_after_training", False):
        splits = ["train", "val"] + (["test"] if cfg.offline_crv.get("test_after_training", True) else [])
        cleanup_feature_caches(cfg, checkpoint_hash, splits, directory, logger)
    print("Complete full-model checkpoints: {}".format(directory))
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--stage", choices=("cache", "train", "all"), default="all")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    validate_config(cfg)
    use_cache = cfg.offline_crv.get("use_cache", True)
    if not use_cache and (args.stage == "cache" or args.splits):
        raise ValueError("use_cache=False runs online frozen inference; --stage cache/--splits require use_cache=True")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run in the configured PropVG CUDA environment")
        torch.cuda.set_device(device)
    set_random_seed(cfg.seed, cfg.get("deterministic", True), cfg.get("strict_reproducibility", True))
    logger = get_root_logger()
    checkpoint_hash = file_sha256(cfg.offline_crv.baseline_checkpoint)
    with feature_cache_lock(cfg.offline_crv.cache_dir) if use_cache else nullcontext():
        if use_cache and args.stage in ("cache", "all"):
            splits = args.splits or (["train", "val", "test"] if cfg.offline_crv.test_after_training else ["train", "val"])
            extract_caches(cfg, splits, checkpoint_hash, device, logger)
        if args.stage in ("train", "all"):
            train_cached(cfg, checkpoint_hash, device)


if __name__ == "__main__":
    try:
        main()
    finally:
        finish_tracking(exit_code=0 if sys.exc_info()[0] is None else 1)
