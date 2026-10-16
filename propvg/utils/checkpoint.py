import torch
import shutil
import os
import os.path as osp
import re
from propvg.utils import is_main
from .logger import get_root_logger
import copy

try:
    import apex
except:
    pass


def is_paral_model(model):
    from mmcv.parallel import MMDistributedDataParallel
    from torch.nn.parallel import DistributedDataParallel

    return isinstance(model, MMDistributedDataParallel) or isinstance(model, DistributedDataParallel)


def is_paral_state(state_dict):
    return list(state_dict.keys())[0].startswith("module.")


def de_parallel(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value
    return new_state_dict


def _optimizer_parameter_names(model, optimizer):
    names = {id(parameter): name[7:] if name.startswith("module.") else name
             for name, parameter in model.named_parameters()}
    return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]


def _load_optimizer_state(model, optimizer, checkpoint, model_state):
    """Resume across removal of inactive branches without shifting Adam states."""
    saved = checkpoint["optimizer"]
    current_names = _optimizer_parameter_names(model, optimizer)
    saved_names = checkpoint.get("optimizer_param_names")
    normalize = lambda name: name[7:] if name.startswith("module.") else name
    if saved_names is not None:
        saved_names = [[normalize(name) for name in group] for group in saved_names]
    elif [len(group["params"]) for group in saved["param_groups"]] == [len(group) for group in current_names]:
        optimizer.load_state_dict(saved)
        return
    else:
        # Older train.py groups parameters by backbone/language/head. Recover
        # the original order from state_dict, using known parameter names only
        # (not buffers or repeated state_dict aliases of shared parameters).
        omitted_owners = {}
        for prefix, module in model.named_modules():
            prefix = normalize(prefix)
            for name in getattr(module, "_omitted_parameter_names", []):
                full_name = prefix + "." + name if prefix else name
                omitted_owners[full_name] = prefix + "." if prefix else ""
        active_names = {name for group in current_names for name in group}
        legacy_order = [normalize(name) for name in model_state
                        if normalize(name) in active_names or normalize(name) in omitted_owners]
        saved_names = []
        for group in current_names:
            members = set(group)
            removed_members = {name for name, owner in omitted_owners.items()
                               if any(active.startswith(owner) for active in group)}
            saved_names.append([name for name in legacy_order if name in members or name in removed_members])

    if len(saved_names) != len(current_names) or len(saved["param_groups"]) != len(current_names):
        raise ValueError("Cannot safely resume optimizer: parameter group layout changed")
    groups, retained_ids = [], set()
    for old_group, names, current in zip(saved["param_groups"], saved_names, current_names):
        if len(names) != len(old_group["params"]):
            raise ValueError("Cannot recover legacy optimizer parameter names; use --finetune-from instead")
        current_set = set(current)
        retained = [(name, index) for name, index in zip(names, old_group["params"]) if name in current_set]
        if [name for name, _ in retained] != current:
            raise ValueError("Cannot safely resume optimizer: active parameter order changed")
        ids = [index for _, index in retained]
        groups.append(dict(old_group, params=ids))
        retained_ids.update(ids)
    optimizer.load_state_dict({"param_groups": groups,
                               "state": {key: value for key, value in saved["state"].items() if key in retained_ids}})


def log_loaded_info(ckpt, load_file, dataset=None):
    logger = get_root_logger()
    log_str = f"loaded checkpoint from {load_file}\n"
    best_d_acc, best_miou = 0.0, 0.0

    if "epoch" and "lr" in ckpt:
        log_str += f"epoch: {ckpt['epoch']+1} lr: {ckpt['lr']:.6f}\n"
    if dataset == "GRefCOCO":
        if "best_F1score" in ckpt:
            log_str += f"best F1score: {ckpt['best_F1score']:.2f}\n"
            best_d_acc = ckpt["best_F1score"]
        if "best_gIoU" in ckpt:
            log_str += f"best gIoU: {ckpt['best_gIoU']:.2f}\n"
            best_miou = ckpt["best_gIoU"]
    elif dataset == "RefZOM":
        if "best_mIoU" in ckpt:
            log_str += f"best best_mIoU: {ckpt['best_mIoU']:.2f}\n"
            best_d_acc = ckpt["best_F1score"]
        if "best_macc" in ckpt:
            log_str += f"best best_macc: {ckpt['best_macc']:.2f}\n"
            best_miou = ckpt["best_macc"]
    elif dataset == "AerialVG":
        best_d_acc = ckpt.get(
            "running_best_top1_acc",
            max(ckpt.get("best_top1_acc", 0.0), ckpt.get("top1_acc", 0.0)),
        )
        best_miou = ckpt.get("best_mIoU_at_top1", ckpt.get("best_mIoU", 0.0))
        log_str += f"best Top-1 Acc@0.5: {best_d_acc:.2f}\n"
        if "best_top5_acc_at_top1" in ckpt:
            log_str += f"Top-5 Acc@0.5 at best Top-1: {ckpt['best_top5_acc_at_top1']:.2f}\n"
        log_str += f"box mIoU at best Top-1: {best_miou:.2f}\n"
    logger.info(log_str)
    return best_d_acc, best_miou


# only for finetuning, if resume from pretraining, use load_checkpoint
def load_pretrained_checkpoint(model, finetune_from=None, amp=False, dataset=None):
    # assert model_ema is None, "We do not use EMA during finetuning."
    start_epoch, best_d_acc, best_miou = -1, 0.0, 0.0
    ckpt = torch.load(finetune_from, map_location=lambda storage, loc: storage.cuda())
    state = ckpt["state_dict"]
    if is_paral_state(state) and not is_paral_model(model):
        state = de_parallel(state)
    state_copy = copy.deepcopy(state)
    # state_copy.pop("lan_enc.embedding.weight")

    # model_seq_embed_dim = model.head.transformer.seq_positional_encoding.embedding.weight.size(
    #     0)
    # state_seq_embed_dim = state_copy["head.transformer.seq_positional_encoding.embedding.weight"].size(
    #     0)
    # # finetuning on RES since pretraining is only performed on REC
    # if model_seq_embed_dim != state_seq_embed_dim:
    #     state_copy.pop("head.transformer.seq_positional_encoding.embedding.weight")
    missing_keys, unexpected_keys = model.load_state_dict(state_copy, strict=False)
    if is_main():
        logger = get_root_logger()
        logger.info("missing keys:{}".format(missing_keys))
        logger.info("unexpected keys:{}".format(unexpected_keys))
    if "amp" in ckpt and amp:
        apex.amp.load_state_dict(ckpt["amp"])
    if is_main():
        best_d_acc, best_miou = log_loaded_info(ckpt, finetune_from, dataset)
    return start_epoch, best_d_acc, best_miou


def load_checkpoint(model, resume_from=None, load_from=None, amp=False, optimizer=None, scheduler=None,
                    dataset=None, return_metadata=False):
    start_epoch, best_d_acc, best_giou, best_oiou = -1, 0.0, 0.0, 0.0
    flag = True
    assert not (resume_from is not None and load_from is not None)
    load_file = resume_from or load_from
    ckpt = torch.load(load_file, map_location=lambda storage, loc: storage.cuda())
    state = ckpt["state_dict"]
    if "ema_state_dict" in ckpt:
        ema_state = ckpt["ema_state_dict"]
        if is_paral_state(ema_state) and not is_paral_model(model):
            ema_state = de_parallel(ema_state)
    if is_paral_state(state) and not is_paral_model(model):
        state = de_parallel(state)
    try:
        model.load_state_dict(state, strict=True)
    except:
        model.load_state_dict(state, strict=False)
        flag = False
    if optimizer is not None and "optimizer" in ckpt:
        _load_optimizer_state(model, optimizer, ckpt, state)
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "amp" in ckpt and amp:
        apex.amp.load_state_dict(ckpt["amp"])

    if "epoch" in ckpt:
        if load_from is None and resume_from is not None:
            start_epoch = ckpt["epoch"]
    if is_main():
        best_d_acc, best_giou = log_loaded_info(ckpt, load_file, dataset)
    result = (start_epoch, best_d_acc, best_giou, flag)
    if return_metadata:
        metadata_keys = (
            "epoch", "top1_acc", "top5_acc", "mIoU",
            "best_top1_acc", "running_best_top1_acc",
            "best_top5_acc_at_top1", "best_mIoU_at_top1",
            "scheduler_state_is_next_epoch",
        )
        metadata = {key: ckpt[key] for key in metadata_keys if key in ckpt}
        return result + (metadata,)
    return result


def _prune_epoch_checkpoints(work_dir, max_keep_ckpts):
    """Keep the highest numbered epoch snapshots; never remove named best/latest."""
    snapshots = []
    with os.scandir(work_dir) as entries:
        for entry in entries:
            match = re.fullmatch(r"epoch_(\d+)\.pth", entry.name)
            if match and entry.is_file(follow_symlinks=False):
                snapshots.append((int(match.group(1)), entry.path))
    for _, path in sorted(snapshots)[:-max_keep_ckpts]:
        os.remove(path)


def save_checkpoint(
    work_dir, interval, model, optimizer, scheduler, checkpoint,
    save_key_metric={"Det": "F1score", "Seg": "gIoU"}, max_keep_ckpts=0,
):
    """Save latest/best plus periodic snapshots; 0 disables snapshot pruning."""
    if isinstance(max_keep_ckpts, bool) or not isinstance(max_keep_ckpts, int) or max_keep_ckpts < 0:
        raise ValueError("max_keep_ckpts must be a nonnegative integer")
    epoch = checkpoint["epoch"] + 1
    logger = get_root_logger()
    use_fp16 = checkpoint.pop("use_fp16", False)
    if use_fp16:
        checkpoint.update({"amp": apex.amp.state_dict()})
    checkpoint.update(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_param_names": _optimizer_parameter_names(model, optimizer),
            "scheduler": scheduler.state_dict(),
            "scheduler_state_is_next_epoch": True,
            "lr": optimizer.param_groups[0]["lr"],
        }
    )
    latest_path = osp.join(work_dir, "latest.pth")
    det_best_name = "top1_best.pth" if save_key_metric.get("Det") == "top1_acc" else "det_best.pth"
    det_best_path = osp.join(work_dir, det_best_name)
    segm_best_path = osp.join(work_dir, "segm_best.pth")
    torch.save(checkpoint, latest_path)
    if is_main():
        logger.info(f"saved epoch {epoch} checkpoint at {latest_path}")
    if interval > 0 and epoch % interval == 0:
        shutil.copyfile(latest_path, osp.join(work_dir, f"epoch_{epoch}.pth"))
    if checkpoint[save_key_metric["Det"]] > checkpoint["best_{}".format(save_key_metric["Det"])]:
        shutil.copyfile(latest_path, det_best_path)
        if is_main():
            logger.info(f"saved epoch {epoch} checkpoint at {det_best_path}")
    seg_metric = save_key_metric.get("Seg")
    if seg_metric is not None and checkpoint[seg_metric] > checkpoint["best_{}".format(seg_metric)]:
        shutil.copyfile(latest_path, segm_best_path)
        if is_main():
            logger.info(f"saved epoch {epoch} checkpoint at {segm_best_path}")
    # Prune only after all current snapshots and best checkpoints save successfully.
    if interval > 0 and epoch % interval == 0 and max_keep_ckpts > 0:
        _prune_epoch_checkpoints(work_dir, max_keep_ckpts)
