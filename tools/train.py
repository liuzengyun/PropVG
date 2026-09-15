import time
import argparse
import os
import os.path as osp
import shutil

# Must be configured before CUDA creates a cuBLAS handle.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch.distributed as dist

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel

from propvg.core import build_optimizer, build_scheduler
from propvg.datasets import build_dataset, build_dataloader
from propvg.models import build_model, ExponentialMovingAverage
from propvg.apis import set_random_seed, train_model, evaluate_model
from propvg.utils import (
    get_root_logger,
    load_checkpoint,
    save_checkpoint,
    load_pretrained_checkpoint,
    is_main,
    init_dist,
)
import wandb

import warnings

warnings.filterwarnings("ignore")

try:
    import apex
except:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="SeqTR-train")
    parser.add_argument("config", help="training configuration file path.")
    parser.add_argument("--debug", default=False, help="training configuration file path.")
    parser.add_argument("--work-dir", help="directory of config file, training logs, and checkpoints.")
    parser.add_argument("--resume-from", help="resume training from the saved .pth checkpoint, only used in training.")
    parser.add_argument("--load-from", help="resume training from the saved .pth checkpoint, only used in training.")
    parser.add_argument(
        "--finetune-from",
        help="finetune from the saved .pth checkpoint, only used after SeqTR has been pre-trained on the merged dadtaset.",
    )
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    return args


def main_worker(cfg):
    cfg.distributed = False
    if cfg.launcher == "pytorch":
        cfg.distributed = True
        init_dist()
    # if cfg.distributed and cfg.dataset == "RRefCOCO":
    #     raise NotImplementedError("Distributed training is not supported for RRefCOCO dataset.")
    cfg.rank, cfg.world_size = get_dist_info()
    if is_main():
        # Keep an exact copy of the model code used by this run so that old
        # experiments remain reproducible after the working tree changes.
        source_dir = osp.abspath(osp.join(osp.dirname(__file__), "..", "propvg"))
        backup_dir = osp.join(cfg.work_dir, "propvg")
        shutil.copytree(source_dir, backup_dir, dirs_exist_ok=True)
        wandb.init(project="propvg-seg-offline", tags=[cfg.tag_name], name=cfg.wandb_name, mode="offline")
        logger = get_root_logger(log_file=osp.join(cfg.work_dir, str(cfg.timestamp) + "_train_log.txt"))
        logger.info(cfg.pretty_text)
        cfg.dump(osp.join(cfg.work_dir, f"{cfg.timestamp}_" + osp.basename(cfg.config)))

    datasets_cfgs = [cfg.data.train]
    if cfg.dataset == "Mixed":
        items = [
            "val_refcoco_unc",
            "val_refcocoplus_unc",
            "val_refcocog_umd",
            "val_referitgame_berkeley",
            "val_flickr30k",
        ]
        for item in items:
            if getattr(cfg.data, item, None):
                datasets_cfgs += [getattr(cfg.data, item)]
    elif cfg.dataset == "MixedSeg":
        items = ["val_refcoco_unc", "val_refcocoplus_unc", "val_refcocog_umd", "val_refcocog_google"]
        for item in items:
            if getattr(cfg.data, item, None):
                datasets_cfgs += [getattr(cfg.data, item)]
    elif cfg.dataset == "RRefCOCO":
        items = ["val_rrefcoco", "val_rrefcoco+", "val_rrefcocog"]
        datasets_cfgs += [cfg.data.val_rrefcoco, cfg.data.val_rrefcocoplus, cfg.data.val_rrefcocog]
    else:
        items = ["val"]
        datasets_cfgs += [cfg.data.val]

    datasets = list(map(build_dataset, datasets_cfgs))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets))

    cfg.model.mask_save_target_dir = cfg.work_dir
    # cfg.model.threshold = cfg.threshold
    # cfg.model.box_threshold = cfg.score_threshold
    model = build_model(cfg.model, word_emb=datasets[0].word_emb, num_token=datasets[0].num_token)
    model = model.cuda()
    train_params = [
        {
            "params": [p for n, p in model.named_parameters() if "vis_enc" in n and p.requires_grad],
            "lr": cfg.optimizer_config.pop("lr_vis_enc"),
        },
        {
            "params": [p for n, p in model.named_parameters() if "lan_enc" in n and p.requires_grad],
            "lr": cfg.optimizer_config.pop("lr_lan_enc"),
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "lan_enc" not in n and "vis_enc" not in n and p.requires_grad
            ],
            "lr": cfg.optimizer_config.lr,
        },
    ]

    optimizer = build_optimizer(cfg.optimizer_config, train_params)
    scheduler = build_scheduler(cfg.scheduler_config, optimizer)

    if cfg.use_fp16:
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True

    if cfg.distributed:
        model = MMDistributedDataParallel(model, device_ids=[cfg.rank], find_unused_parameters=True)
    (
        start_epoch,
        best_F1score,
        best_Nacc_rec,
        best_gIoU,
        best_cIoU,
        best_mRR,
        best_rIoU,
        best_mIoU,
        best_oIoU,
        best_macc,
        best_d_acc,
    ) = (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    resume_metadata = {}
    if cfg.resume_from:
        start_epoch, best_d_acc, best_mIoU, flag, resume_metadata = load_checkpoint(
            model, cfg.resume_from, amp=cfg.use_fp16, optimizer=optimizer, scheduler=scheduler,
            dataset=cfg.dataset, return_metadata=True
        )
        # Checkpoints produced by older versions were saved before the epoch's
        # scheduler step. Advance them once so resume matches uninterrupted
        # training. New checkpoints explicitly record the corrected state.
        if not resume_metadata.get("scheduler_state_is_next_epoch", False):
            scheduler.step()
    elif cfg.finetune_from:
        load_pretrained_checkpoint(model, cfg.finetune_from, amp=cfg.use_fp16, dataset=cfg.dataset)
    elif cfg.load_from:
        start_epoch, best_F1score, best_gIoU, flag = load_checkpoint(
            model, load_from=cfg.load_from, dataset=cfg.dataset
        )

    import time

    begin_time = time.time()
    # Top-5 and mIoU in the final directory name must come from the epoch with
    # the best Top-1 accuracy, rather than independently chosen/final values.
    best_aerialvg_metrics = None
    if cfg.dataset == "AerialVG" and resume_metadata:
        running_top1 = resume_metadata.get(
            "running_best_top1_acc",
            max(resume_metadata.get("best_top1_acc", 0.0), resume_metadata.get("top1_acc", 0.0)),
        )
        # New checkpoints store all three metrics from the same best-Top1
        # epoch. The fallback keeps older checkpoints resumable.
        if "best_top5_acc_at_top1" in resume_metadata:
            best_aerialvg_metrics = (
                running_top1,
                resume_metadata["best_top5_acc_at_top1"],
                resume_metadata["best_mIoU_at_top1"],
            )
        elif resume_metadata.get("top1_acc", 0.0) >= resume_metadata.get("best_top1_acc", 0.0):
            best_aerialvg_metrics = (
                resume_metadata.get("top1_acc", 0.0),
                resume_metadata.get("top5_acc", 0.0),
                resume_metadata.get("mIoU", 0.0),
            )
        else:
            best_aerialvg_metrics = (running_top1, 0.0, 0.0)
        best_d_acc = running_top1
        best_mIoU = max(best_mIoU, best_aerialvg_metrics[2])

        # Runs are written to a fresh timestamp directory. Carry the previous
        # best checkpoint forward so a resumed run with no new best still has
        # the correct model for final single-GPU testing.
        if is_main():
            previous_best = osp.join(osp.dirname(cfg.resume_from), "top1_best.pth")
            current_best = osp.join(cfg.work_dir, "top1_best.pth")
            if osp.isfile(previous_best):
                shutil.copyfile(previous_best, current_best)
            elif resume_metadata.get("top1_acc", -1.0) >= running_top1:
                shutil.copyfile(cfg.resume_from, current_best)
            else:
                logger.warning(
                    "Could not locate the historical top1_best.pth beside %s; "
                    "final best-checkpoint testing will require a new best epoch.",
                    cfg.resume_from,
                )
    for epoch in range(start_epoch + 1, cfg.scheduler_config.max_epoch):
        start_time = time.time()
        train_model(epoch, cfg, model, optimizer, dataloaders[0])
        # Save the scheduler state for the *next* epoch. This keeps resumed LR
        # progression identical to uninterrupted training.
        scheduler.step()
        this_epoch_train_time = int(time.time() - start_time)
        if is_main():
            logger.info("this_epoch_train_time={}m-{}s".format(this_epoch_train_time // 60, this_epoch_train_time % 60))

        if epoch % cfg.evaluate_interval == 0 and epoch >= cfg.start_evaluate_epoch:
            F1score, Nacc_rec, gIoU, cIoU = 0, 0, 0, 0
            mIoU, oIoU, macc = 0, 0, 0
            mRR, rIoU = 0, 0
            d_acc, top5_acc = 0, 0

            for _prefix, _loader in zip(items, dataloaders[1:]):
                if is_main():
                    logger.info("Evaluating dataset: {}".format(_loader.dataset.which_set))
                if cfg.dataset == "GRefCOCO":
                    set_F1score, set_Nacc_rec, set_T_acc_rec, set_gIoU, set_cIoU, set_PR_list_res = evaluate_model(
                        _prefix, cfg, model, _loader
                    )
                    F1score += set_F1score
                    Nacc_rec += set_Nacc_rec
                    gIoU += set_gIoU
                    cIoU += set_cIoU
                elif cfg.dataset == "RefZOM":
                    set_mIoU, set_oIoU, set_macc, set_PR_list_res = evaluate_model(_prefix, cfg, model, _loader)
                    mIoU += set_mIoU
                    oIoU += set_oIoU
                    macc += set_macc
                elif cfg.dataset == "RRefCOCO":
                    # set_mIoU, set_oIoU, set_mRR, set_rIoU = evaluate_model(_prefix, cfg, model, _loader)
                    set_mIoU, set_oIoU, set_mRR, set_rIoU = 0, 0, 0, 0
                    mIoU += set_mIoU
                    oIoU += set_oIoU
                    mRR += set_mRR
                    rIoU += set_rIoU
                elif cfg.dataset == "AerialVG":
                    d_acc, top5_acc, mIoU = evaluate_model(_prefix, cfg, model, _loader)
                    if best_aerialvg_metrics is None or d_acc > best_aerialvg_metrics[0]:
                        best_aerialvg_metrics = (d_acc, top5_acc, mIoU)
                elif (
                    cfg.dataset == "MixedSeg"
                    or cfg.dataset == "RefCOCOUNC"
                    or cfg.dataset == "RefCOCOPlusUNC"
                    or cfg.dataset == "RefCOCOgUMD"
                ):
                    set_det_acc, set_mIoU, set_oIoU, set_macc, set_det_accs = evaluate_model(
                        _prefix, cfg, model, _loader
                    )
                    d_acc += set_det_acc
                    mIoU += set_mIoU
                    oIoU += set_oIoU

            if cfg.dataset == "GRefCOCO":
                F1score /= len(dataloaders[1:])
                Nacc_rec /= len(dataloaders[1:])
                gIoU /= len(dataloaders[1:])
                cIoU /= len(dataloaders[1:])
            elif cfg.dataset == "RefZOM":
                mIoU /= len(dataloaders[1:])
                oIoU /= len(dataloaders[1:])
                macc /= len(dataloaders[1:])
            elif cfg.dataset == "RRefCOCO":
                mIoU /= len(dataloaders[1:])
                oIoU /= len(dataloaders[1:])
                mRR /= len(dataloaders[1:])
                rIoU /= len(dataloaders[1:])
            elif cfg.dataset == "AerialVG":
                pass
            elif (
                cfg.dataset == "MixedSeg"
                or cfg.dataset == "RefCOCOUNC"
                or cfg.dataset == "RefCOCOPlusUNC"
                or cfg.dataset == "RefCOCOgUMD"
            ):
                mIoU /= len(dataloaders[1:])
                oIoU /= len(dataloaders[1:])
                d_acc /= len(dataloaders[1:])

            if is_main():
                this_epoch_total_time = int(time.time() - start_time)
                logger.info(
                    "this_epoch_total_time={}m-{}s".format(this_epoch_total_time // 60, this_epoch_total_time % 60)
                )
                total_time = int(time.time() - begin_time)
                logger.info("total_time={}m-{}s".format(total_time // 60, total_time % 60))

            if is_main() and epoch >= cfg.start_save_checkpoint:
                if cfg.dataset == "GRefCOCO":
                    saved_info = {
                        "epoch": epoch,
                        "F1score": F1score,
                        "Nacc_rec": Nacc_rec,
                        "gIoU": gIoU,
                        "cIoU": cIoU,
                        "best_F1score": best_F1score,
                        "best_Nacc_rec": best_Nacc_rec,
                        "best_gIoU": best_gIoU,
                        "best_cIoU": best_cIoU,
                    }
                    save_compare_det = "F1score"
                    save_compare_seg = "cIoU"
                elif cfg.dataset == "RefZOM":
                    saved_info = {
                        "epoch": epoch,
                        "mIoU": mIoU,
                        "oIoU": oIoU,
                        "macc": macc,
                        "best_mIoU": best_mIoU,
                        "best_oIoU": best_oIoU,
                        "best_macc": best_macc,
                    }
                    save_compare_det = "mIoU"
                    save_compare_seg = "macc"
                elif cfg.dataset == "RRefCOCO":
                    saved_info = {
                        "epoch": epoch,
                        "mIoU": mIoU,
                        "oIoU": oIoU,
                        "mRR": mRR,
                        "rIoU": rIoU,
                        "best_mIoU": best_mIoU,
                        "best_oIoU": best_oIoU,
                        "best_mRR": best_mRR,
                        "best_rIoU": best_rIoU,
                    }
                    save_compare_det = "mIoU"
                    save_compare_seg = "mRR"
                elif cfg.dataset == "AerialVG":
                    saved_info = {
                        "epoch": epoch,
                        "top1_acc": d_acc,
                        "top5_acc": top5_acc,
                        "mIoU": mIoU,
                        "best_top1_acc": best_d_acc,
                        "best_mIoU": best_mIoU,
                        "running_best_top1_acc": best_aerialvg_metrics[0],
                        "best_top5_acc_at_top1": best_aerialvg_metrics[1],
                        "best_mIoU_at_top1": best_aerialvg_metrics[2],
                    }
                    save_compare_det = "top1_acc"
                    save_compare_seg = None
                elif (
                    cfg.dataset == "MixedSeg"
                    or cfg.dataset == "RefCOCOUNC"
                    or cfg.dataset == "RefCOCOPlusUNC"
                    or cfg.dataset == "RefCOCOgUMD"
                ):
                    saved_info = {
                        "epoch": epoch,
                        "mIoU": mIoU,
                        "oIoU": oIoU,
                        "d_acc": d_acc,
                        "best_d_acc": best_d_acc,
                        "best_mIoU": best_mIoU,
                        "best_oIoU": best_oIoU,
                    }
                    save_compare_det = "d_acc"
                    save_compare_seg = "mIoU"

                save_checkpoint(
                    cfg.work_dir,
                    cfg.save_interval,
                    model,
                    optimizer,
                    scheduler,
                    saved_info,
                    save_key_metric={"Det": save_compare_det, "Seg": save_compare_seg},
                )
            if cfg.dataset == "GRefCOCO":
                best_F1score = max(F1score, best_F1score)
                best_gIoU = max(gIoU, best_gIoU)
                best_cIoU = max(cIoU, best_cIoU)
            elif cfg.dataset == "RefZOM":
                best_mIoU = max(mIoU, best_mIoU)
                best_oIoU = max(oIoU, best_oIoU)
                best_macc = max(macc, best_macc)
            elif cfg.dataset == "RRefCOCO":
                best_mIoU = max(mIoU, best_mIoU)
                best_oIoU = max(oIoU, best_oIoU)
                best_mRR = max(mRR, best_mRR)
                best_rIoU = max(rIoU, best_rIoU)
            elif cfg.dataset == "AerialVG":
                best_d_acc = max(d_acc, best_d_acc)
                best_mIoU = max(mIoU, best_mIoU)
            elif (
                cfg.dataset == "MixedSeg"
                or cfg.dataset == "RefCOCOUNC"
                or cfg.dataset == "RefCOCOPlusUNC"
                or cfg.dataset == "RefCOCOgUMD"
            ):
                best_mIoU = max(mIoU, best_mIoU)
                best_oIoU = max(oIoU, best_oIoU)
                best_d_acc = max(d_acc, best_d_acc)

        if cfg.distributed:
            dist.barrier()

    main_process = is_main()
    if cfg.distributed:
        dist.destroy_process_group()

    test_aerialvg_metrics = None
    if main_process and cfg.dataset == "AerialVG" and best_aerialvg_metrics is not None:
        # Final test is deliberately single-GPU and is performed only once,
        # after model selection has finished on val. This avoids distributed
        # sampler padding and prevents test results from influencing which
        # checkpoint is selected.
        best_checkpoint = osp.join(cfg.work_dir, "top1_best.pth")
        if not osp.isfile(best_checkpoint):
            raise FileNotFoundError(f"Best AerialVG checkpoint not found: {best_checkpoint}")

        eval_model = model.module if hasattr(model, "module") else model
        load_checkpoint(eval_model, load_from=best_checkpoint, dataset=cfg.dataset)

        single_gpu_cfg = cfg.copy()
        single_gpu_cfg.distributed = False
        single_gpu_cfg.rank = 0
        single_gpu_cfg.world_size = 1
        test_dataset = build_dataset(single_gpu_cfg.data.test)
        test_loader = build_dataloader(single_gpu_cfg, test_dataset)
        logger.info("Evaluating best-val checkpoint on AerialVG-test with one GPU: %s", best_checkpoint)
        test_aerialvg_metrics = evaluate_model("test", single_gpu_cfg, eval_model, test_loader)

        val_top1, val_top5, val_miou = best_aerialvg_metrics
        test_top1, test_top5, test_miou = test_aerialvg_metrics
        final_work_dir = (
            f"{cfg.work_dir}"
            f"+val_top1={val_top1:.2f}+val_top5={val_top5:.2f}+val_miou={val_miou:.2f}"
            f"+test_top1={test_top1:.2f}+test_top5={test_top5:.2f}+test_miou={test_miou:.2f}"
        )
        if osp.abspath(final_work_dir) != osp.abspath(cfg.work_dir):
            if osp.exists(final_work_dir):
                raise FileExistsError(f"Cannot rename work directory: destination already exists: {final_work_dir}")
            os.rename(cfg.work_dir, final_work_dir)

    wandb.finish()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # cfg.work_dir = f"./work_dir/{cfg.timestamp}_" + osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = f"./work_dir/" + args.config.split("configs/")[-1].split(".py")[0]
    cfg.work_dir = osp.join(cfg.work_dir, f"{cfg.timestamp}")
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.finetune_from is not None:
        cfg.finetune_from = args.finetune_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    cfg.launcher = args.launcher
    cfg.config = args.config
    cfg.debug = args.debug
    cfg.wandb_name = "-".join(cfg.work_dir.split("/")[3:-1])
    cfg.tag_name = "-".join(cfg.work_dir.split("/")[3:-2])
    cfg.score_threshold = cfg.model.post_params["score_threshold"]

    if is_main():
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if cfg.seed is not None:
        set_random_seed(
            cfg.seed,
            deterministic=cfg.deterministic,
            strict=cfg.get("strict_reproducibility", False),
        )

    main_worker(cfg)


if __name__ == "__main__":
    main()
