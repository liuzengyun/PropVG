import time
import copy
import os
import numpy
import torch
import random

from propvg.apis.test import grec_evaluate_f1_nacc, gres_evaluate

from .test import accuracy
from propvg.datasets import extract_data
from propvg.utils import get_root_logger, reduce_mean, is_main
from propvg.utils.experiment_tracking import log_metrics, set_tracking_context

try:
    import apex
except:
    pass


def set_random_seed(seed, deterministic=False, strict=False):
    """Args:
    seed (int): Seed to be used.
    deterministic (bool): Whether to set the deterministic option for
        CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
        to True and `torch.backends.cudnn.benchmark` to False.
        Default: False.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if strict:
        # Use deterministic kernels whenever PyTorch provides them. Some
        # third-party CUDA extensions used by Deformable-DETR are outside
        # PyTorch's determinism checks, so warn instead of aborting training.
        torch.use_deterministic_algorithms(True, warn_only=True)


def clip_training_gradients(model, max_norm):
    """Independent heads must not change the base model's clipping norm."""
    base_model = model.module if hasattr(model, "module") else model
    head = getattr(base_model, "head", base_model)
    independent = [branch for branch in (getattr(head, "crv", None), getattr(head, "correctness", None))
                   if branch is not None]
    relation = getattr(head, "relation_refiner", None)
    birs = getattr(head, "birs", None)
    if birs is not None and birs.detach_inputs:
        independent.append(birs)
    if relation is not None and relation.detach_inputs:
        # Detaching the loss is insufficient on its own: including new head
        # gradients in a global norm would still change the CRS update scale.
        independent.append(relation)
    if not independent:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    independent_parameters = [list(branch.parameters()) for branch in independent]
    independent_ids = {id(parameter) for parameters in independent_parameters for parameter in parameters}
    base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in independent_ids]
    torch.nn.utils.clip_grad_norm_(base_parameters, max_norm)
    for parameters in independent_parameters:
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return norm


def train_model(epoch, cfg, model, optimizer, loader):
    model.train()

    if cfg.distributed:
        loader.sampler.set_epoch(epoch)

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    loss_det_list, loss_mask_list, loss_label_nt_list, loss_refer_list, loss_filter_list = [], [], [], [], []
    crv_history = {}
    iteration_times, data_times = [], []
    for batch, inputs in enumerate(loader):
        data_time = time.time() - end
        if not cfg.distributed:
            inputs = extract_data(inputs)

        losses, predictions = model(**inputs, epoch=epoch, rescale=False)

        loss_det = losses.pop("loss_det", torch.tensor([0.0], device=device))
        loss_mask = losses.pop("loss_mask", torch.tensor([0.0], device=device))
        loss_label_nt = losses.pop("loss_label_nt", torch.tensor([0.0], device=device))
        loss_refer = losses.pop("loss_refer", torch.tensor([0.0], device=device))
        loss_filter = losses.pop("loss_filter", torch.tensor([0.0], device=device))
        loss = loss_det + loss_mask + loss_label_nt + loss_refer + loss_filter
        crv_values = {}
        if "loss_crv_cf" in losses:
            # Preserve old CRV fields while collecting optional new losses and
            # diagnostics; CRV-disabled baseline logging is unchanged.
            for key in tuple(losses):
                if key.startswith("loss_crv_"):
                    value = losses.pop(key)
                    loss = loss + value
                    crv_values[key] = value.detach()
                elif key.startswith("metric_crv_"):
                    crv_values[key[len("metric_"):]] = losses.pop(key).detach()
        correctness_values = {}
        birs_values = {}
        if "loss_birs_final" in losses:
            for key in tuple(losses):
                if key.startswith("loss_birs_"):
                    value = losses.pop(key)
                    loss = loss + value
                    birs_values[key] = value.detach()
                elif key.startswith("metric_birs_"):
                    birs_values[key[len("metric_"):]] = losses.pop(key).detach()
        relation_values = {}
        if "loss_relation_final" in losses:
            value = losses.pop("loss_relation_final")
            loss = loss + value
            relation_values["loss_relation_final"] = value.detach()
            for key in tuple(losses):
                if key.startswith("metric_relation_refinement_"):
                    relation_values[key[len("metric_"):]] = losses.pop(key).detach()
        if "loss_correctness" in losses:
            value = losses.pop("loss_correctness")
            loss = loss + value
            correctness_values["loss_correctness"] = value.detach()
            for key in tuple(losses):
                if key.startswith("metric_correctness_"):
                    correctness_values[key[len("metric_"):]] = losses.pop(key).detach()
        # loss = loss_det
        optimizer.zero_grad()
        if cfg.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if cfg.grad_norm_clip:
            clip_training_gradients(model, cfg.grad_norm_clip)
        optimizer.step()
        if birs_values and birs_values["birs_supervised_samples"].item() > 0:
            base_model = model.module if hasattr(model, "module") else model
            base_model.head.birs.mark_optimizer_step()
        crv_values.update(birs_values)
        if correctness_values and correctness_values["correctness_valid_samples"].item() > 0:
            base_model = model.module if hasattr(model, "module") else model
            base_model.head.correctness.mark_optimizer_step()
        crv_values.update(correctness_values)
        if relation_values and relation_values["relation_refinement_supervised_samples"].item() > 0:
            base_model = model.module if hasattr(model, "module") else model
            base_model.head.relation_refiner.mark_optimizer_step()
        crv_values.update(relation_values)

        if cfg.distributed:
            loss_det = reduce_mean(loss_det)
            loss_mask = reduce_mean(loss_mask)
            loss_label_nt = reduce_mean(loss_label_nt)
            loss_refer = reduce_mean(loss_refer)
            loss_filter = reduce_mean(loss_filter)
        loss_det_list.append(loss_det.item())
        loss_mask_list.append(loss_mask.item())
        loss_label_nt_list.append(loss_label_nt.item())
        loss_refer_list.append(loss_refer.item())
        loss_filter_list.append(loss_filter.item())
        for key, value in crv_values.items():
            if cfg.distributed:
                value = reduce_mean(value)
            crv_history.setdefault(key, []).append(value.item())

        iteration_times.append(time.time() - end)
        data_times.append(data_time)

        if is_main():
            if (batch + 1) % cfg.log_interval == 0 or batch + 1 == batches:
                logger = get_root_logger()
                logger.info(
                    f"train - epoch [{epoch+1}]-[{batch+1}/{batches}] "
                    + f"time: {(time.time()- end):.2f}, data_time: {data_time:.2f}, "
                    + f"loss_det: {sum(loss_det_list) / len(loss_det_list) :.4f}, "
                    + f"loss_mask: {sum(loss_mask_list) / len(loss_mask_list):.4f}, "
                    + f"loss_nt: {sum(loss_label_nt_list) / len(loss_label_nt_list):.4f}, "
                    + f"loss_refer: {sum(loss_refer_list) / len(loss_refer_list):.4f}, "
                    + f"loss_filter: {sum(loss_filter_list) / len(loss_filter_list):.4f}, "
                    + f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                    + "".join(f"{key}: {sum(values) / len(values):.4f}, " for key, values in crv_history.items())
                )

                values = {
                        "loss_det": sum(loss_det_list) / len(loss_det_list),
                        "loss_mask": sum(loss_mask_list) / len(loss_mask_list),
                        "loss_label_nt": sum(loss_label_nt_list) / len(loss_label_nt_list),
                        "loss_refer": sum(loss_refer_list) / len(loss_refer_list),
                        "loss_filter": sum(loss_filter_list) / len(loss_filter_list),
                        "lr": optimizer.param_groups[0]["lr"],
                        **{key: sum(values) / len(values) for key, values in crv_history.items()},
                    }
                set_tracking_context(epoch=epoch + 1, global_step=epoch * batches + batch + 1)
                # Reuse already-reduced CPU values: no new model forward,
                # tensor transfer, gradient scan or distributed collective.
                history = dict(loss_det=loss_det_list, loss_mask=loss_mask_list,
                               loss_label_nt=loss_label_nt_list, loss_refer=loss_refer_list,
                               loss_filter=loss_filter_list, **crv_history)
                window = batch % cfg.log_interval + 1
                metrics = dict(values)  # Retain old chart names for comparisons.
                for key, series in history.items():
                    metrics["train/" + key] = sum(series) / len(series)
                    metrics["train_current/" + key] = series[-1]
                    metrics["train_window/" + key] = sum(series[-window:]) / window
                metrics["train/loss_total"] = sum(value for key, value in values.items() if key.startswith("loss_"))
                metrics["time/iteration_seconds"] = sum(iteration_times[-window:]) / window
                metrics["time/data_seconds"] = sum(data_times[-window:]) / window
                nominal_batch = cfg.data.samples_per_gpu * (cfg.world_size if cfg.distributed else 1)
                metrics["perf/nominal_images_per_second"] = nominal_batch / max(metrics["time/iteration_seconds"], 1e-8)
                for group_index, group in enumerate(optimizer.param_groups):
                    group_name = ("backbone", "language", "head")[group_index] if len(optimizer.param_groups) == 3 else str(group_index)
                    metrics["lr/" + group_name] = group["lr"]
                if device.type == "cuda":
                    metrics["memory/allocated_mib"] = torch.cuda.memory_allocated(device) / 2**20
                    metrics["memory/reserved_mib"] = torch.cuda.memory_reserved(device) / 2**20
                    metrics["memory/peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
                log_metrics(metrics)

        end = time.time()
