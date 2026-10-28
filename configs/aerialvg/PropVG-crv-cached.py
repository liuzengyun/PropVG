"""Frozen PropVG-BASE -> cached OR online features -> CRV-only training.

Run with tools/train_crv_cached.py, NOT tools/train.py/dist_train.sh.
This is a complete independent config; no inheritance of the user's run config.
The final full checkpoint is evaluated by the existing tools/eval_aerialvg.py.
"""
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
del AerialVG, LoadAerialVGAnnotations

work_dir = "/media/vision/lzy/PropVG/work_dir/aerialvg/PropVG-crv-cached"
dataset = "AerialVG"
dataset_type = "AerialVG"
seed = 6666
deterministic = True
strict_reproducibility = True
log_level = "INFO"
log_interval = 50
distributed = False
rank = 0
world_size = 1
use_fp16 = False
ema = False
resume_from = None
load_from = None
finetune_from = None

# Source: the original fully trained BASE experiment, not the current Grab run.
# When changing the baseline, match its architecture/image size/switches below.
baseline_checkpoint = (
    "/media/vision/lzy/PropVG/work_dir/aerialvg/PropVG-BASE/"
    "20261015_215412+val_top1=66.49+val_top5=88.06+val_miou=57.81"
    "+test_top1=65.72+test_top5=88.31+test_miou=56.96/top1_best.pth"
)
offline_crv = dict(
    baseline_checkpoint=baseline_checkpoint,
    # None: use embedded BCEM config or discover a single saved config beside
    # the checkpoint. Set this to the ORIGINAL snapshot if discovery is ambiguous.
    baseline_config=None,
    use_cache=True,  # False: run frozen grounding once per batch, no cache I/O.
    cleanup_cache_after_training=True,  # Delete ONLY used feature caches after successful save/eval.
    cache_dir="/media/vision/lzy/PropVG/temp/crv_cache/PropVG-BASE-20261015-fp32",
    cache_dtype="float32",  # Exact FP32 features; float16 reduces ONLY V/T/Q storage (approximate).
    extract_batch_size=2,  # Frozen grounding inference, once per image-expression pair.
    extract_workers=4,
    batch_size=1,          # Online mode must also fit the frozen grounding model.
    workers=4,
    max_epochs=20,
    lr=0.0001,
    grad_clip=0.15,
    test_after_training=True,  # Test once, selecting the checkpoint ONLY by val Top-1.
    backup_code=True,
    rename_with_metrics=True,
)

max_token = 64
img_size = 586
patch_size = 16
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
# No proposal training is performed. Only regions[0] is read as the refer GT.
teacher = False
use_auxiliary_gt = False
enable_grab = False
enable_bcem = False
enable_mask = False
# Optional frozen observer already trained IN baseline_checkpoint; not trained
# here and never used to route/fuse CRV. The default BASE checkpoint lacks it.
enable_correctness = False
correctness = dict(hidden_dim=128, topk=5, duplicate_iou=0.8,
                   loss_weight=1.0, decision_threshold=0.5)
enable_crv = True
freeze_beit3 = True
crv_topk = 5  # Only reorder baseline Top-5; preserve Top-5 candidate membership.
crv_grid_size = 48
crv_uncertainty_threshold = 0.6
crv_start_epoch = 1

pipeline = [
    dict(type="LoadAerialVGAnnotations", max_token=max_token,
         use_auxiliary_boxes=False, use_teacher_boxes=False),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=True),
    dict(type="Pad", size=(img_size, img_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "text_attention_mask", "gt_bbox"],
         meta_keys=["filename", "expression", "ori_shape", "img_shape", "pad_shape", "scale_factor",
                    "refer_target_index", "empty"]),
]
data = dict(
    samples_per_gpu=1, workers_per_gpu=4,
    train=dict(type=dataset_type, which_set="train", imgsfile="data/AerialVG/images",
               annsfile="data/AerialVG/annotation/vg_train_odvg.jsonl", pipeline=pipeline),
    val=dict(type=dataset_type, which_set="val", imgsfile="data/AerialVG/images",
             annsfile="data/AerialVG/annotation/vg_val_odvg.jsonl", pipeline=pipeline),
    test=dict(type=dataset_type, which_set="test", imgsfile="data/AerialVG/images",
              annsfile="data/AerialVG/annotation/vg_test_odvg.jsonl", pipeline=pipeline),
)

model = dict(
    type="MIXRefUniModel_OMG",
    # A complete baseline checkpoint supplies BEiT-3; no separate pretrained download/load.
    vis_enc=dict(type="BEIT3", img_size=img_size, patch_size=patch_size, vit_type="base",
                 drop_path_rate=0.1, vocab_size=64010, freeze_layer=-1, freeze_beit3=freeze_beit3,
                 vision_embed_proj_interpolate=False, pretrain=None),
    lan_enc=None, fusion=None,
    head=dict(
        type="REFHead", input_channels=768, hidden_channels=256, num_queries=256,
        detr_loss=dict(criterion=dict(loss_class=1., loss_bbox=5., loss_giou=2.),
                       matcher=dict(cost_class=1., cost_bbox=5., cost_giou=2.)),
        loss_weight=dict(mask=dict(dice=1., bce=1., nt=0.2, neg=0), bbox=0.1, allbbox=0.1, refer=1.),
        MTD=dict(K=250), enable_grab=enable_grab, enable_mask=enable_mask,
        enable_bcem=enable_bcem,
        # Defined for architecture portability, but not constructed for BASE.
        bcem=dict(context_dim=128, num_heads=4, num_sectors=4, sampling_mode="ray", num_rays=16,
                  radial_edges=(0., 0.10, 0.25, 0.50, 1.), angular_samples=5, radial_samples=4,
                  min_valid_points=2, context_strength=0.1, chunk_size=32,
                  use_position=True, use_transitions=True, use_gru=True, use_text_attention=True,
                  use_sparse_selection=True, learnable_gate=True, checkpoint_chunks=True),
        enable_correctness=enable_correctness, correctness=correctness,
        enable_crv=enable_crv,
        crv=dict(topk=crv_topk, uncertainty_topn=5, grid_size=crv_grid_size, start_epoch=crv_start_epoch,
                 scorer_dim=128, scorer_heads=4, scorer_layers=2, chunk_size=8,
                 inner_scale=1.5, outer_scale=2., duplicate_iou=0.8,
                 uncertainty_temperature=1., uncertainty_threshold=crv_uncertainty_threshold,
                 gate_initial_probability=0.5, cf_temperature=1.,
                 negative_gt_coverage=0.01, gt_loss_weight=0.5,
                 cf_loss_weight=1., fusion_loss_weight=1., gate_loss_weight=1., gate_dim=64,
                 calibration_eps=1e-4, calibration_min_scale=0.1, positive_margin=1.,
                 ranking_margin=0.5, safety_margin=0.5, safety_weight=2., repair_weight=2.),
    ),
    post_params=dict(score_weighted=False, mask_threshold=0.5, score_threshold=0.7,
                     with_nms=False, with_mask=False),
    process_visual=False,
)
swanlab = dict(mode="online", project="propvg", init_timeout=30, finish_wait=1, stats_interval=30)
