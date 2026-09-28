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
from collections import defaultdict
import wandb

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


def train_model(epoch, cfg, model, optimizer, loader):
    model.train()

    if cfg.distributed:
        loader.sampler.set_epoch(epoch)

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    loss_det_list, loss_mask_list, loss_label_nt_list, loss_refer_list, loss_filter_list, loss_crv_list = [], [], [], [], [], []
    crv_cf_list, crv_final_list = [], []
    crv_alpha_list, crv_pair_list = [], []
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
        loss_crv = losses.pop("loss_crv", torch.tensor([0.0], device=device))
        crv_cf = losses.pop("metric_crv_cf", torch.tensor([0.0], device=device))
        crv_final = losses.pop("metric_crv_final", torch.tensor([0.0], device=device))
        crv_alpha = losses.pop("metric_crv_alpha", torch.tensor([0.0], device=device))
        crv_pairs = losses.pop("metric_crv_pairs", torch.tensor([0.0], device=device))
        loss = loss_det + loss_mask + loss_label_nt + loss_refer + loss_filter + loss_crv
        # loss = loss_det
        optimizer.zero_grad()
        if cfg.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if cfg.grad_norm_clip:
            # Independent CRV gradients must not change baseline clipping.
            crv_parameters, base_parameters = [], []
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if ".crv." in name:
                    crv_parameters.append(parameter)
                else:
                    base_parameters.append(parameter)
            if base_parameters:
                torch.nn.utils.clip_grad_norm_(base_parameters, cfg.grad_norm_clip)
            if crv_parameters:
                torch.nn.utils.clip_grad_norm_(crv_parameters, cfg.grad_norm_clip)
        optimizer.step()

        if cfg.distributed:
            loss_det = reduce_mean(loss_det)
            loss_mask = reduce_mean(loss_mask)
            loss_label_nt = reduce_mean(loss_label_nt)
            loss_refer = reduce_mean(loss_refer)
            loss_filter = reduce_mean(loss_filter)
            loss_crv = reduce_mean(loss_crv)
            crv_cf = reduce_mean(crv_cf)
            crv_final = reduce_mean(crv_final)
            crv_alpha = reduce_mean(crv_alpha)
            crv_pairs = reduce_mean(crv_pairs)
        loss_det_list.append(loss_det.item())
        loss_mask_list.append(loss_mask.item())
        loss_label_nt_list.append(loss_label_nt.item())
        loss_refer_list.append(loss_refer.item())
        loss_filter_list.append(loss_filter.item())
        loss_crv_list.append(loss_crv.item())
        crv_cf_list.append(crv_cf.item())
        crv_final_list.append(crv_final.item())
        crv_alpha_list.append(crv_alpha.item())
        crv_pair_list.append(crv_pairs.item())

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
                    + f"loss_crv: {sum(loss_crv_list) / len(loss_crv_list):.4f}, "
                    + f"crv_cf: {sum(crv_cf_list) / len(crv_cf_list):.4f}, "
                    + f"crv_final: {sum(crv_final_list) / len(crv_final_list):.4f}, "
                    + f"crv_alpha: {sum(crv_alpha_list) / len(crv_alpha_list):.4f}, "
                    + f"crv_pairs: {sum(crv_pair_list) / len(crv_pair_list):.4f}, "
                    + f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
                )

                wandb.log(
                    {
                        "loss_det": sum(loss_det_list) / len(loss_det_list),
                        "loss_mask": sum(loss_mask_list) / len(loss_mask_list),
                        "loss_label_nt": sum(loss_label_nt_list) / len(loss_label_nt_list),
                        "loss_refer": sum(loss_refer_list) / len(loss_refer_list),
                        "loss_filter": sum(loss_filter_list) / len(loss_filter_list),
                        "loss_crv": sum(loss_crv_list) / len(loss_crv_list),
                        "crv_cf": sum(crv_cf_list) / len(crv_cf_list),
                        "crv_final": sum(crv_final_list) / len(crv_final_list),
                        "crv_alpha": sum(crv_alpha_list) / len(crv_alpha_list),
                        "crv_pairs": sum(crv_pair_list) / len(crv_pair_list),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

        end = time.time()
