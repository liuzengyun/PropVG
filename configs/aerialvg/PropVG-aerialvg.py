_base_ = ["../_base_/misc.py"]
# Register the dataset-local components when this config is loaded.  This
# keeps the original train.py usable for both single- and multi-GPU runs.
from propvg.datasets.aerialvg import AerialVG  # noqa: F401
from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
del AerialVG, LoadAerialVGAnnotations

work_dir = "/media/vision/lzy/PropVG/work_dir/aerialvg/PropVG-crv1116"

# AerialVG provides one referred target (regions[0]); optional additional
# region boxes serve only as proposal/teacher-style supervision.
# Select the AerialVG-specific validation branch in the training runner.
dataset = "AerialVG"
dataset_type = "AerialVG"
# Maximum observed AerialVG caption length is 55 SentencePiece tokens;
# 64 leaves room for BEiT3 BOS/EOS tokens without truncation.
max_token = 64

# Fix every controllable Python/NumPy/PyTorch/DataLoader random source.
strict_reproducibility = True

# False: supervise only regions[0] (the referring target).
# True: also use regions[1:] as DETR proposal detection targets.
use_auxiliary_gt = False
# True: keep regions[0] as the referring GT and append retained predictions
# from the configured teacher JSON as proposal-only detection supervision.
# Validation and test never load teacher predictions.
teacher = True
# The teacher file must match the configured training annotation records by
# sample_id and filename. Validation and test do not read it.
teacher_annsfile = "data/AerialVG/teacher_trainset.json"
# Maximum number of extra teacher boxes per training sample after refer-GT
# deduplication and teacher NMS. 0 uses only the original refer GT; None keeps
# every available teacher box. The current JSON contains exactly 5 raw boxes
# per sample, so values above 5 cannot add supervision with this file.
teacher_max_boxes = 5
teacher_max10 = True  # Filter teacher boxes whose clipped pixel area is >10x the original refer GT.
# Teacher predictions overlapping the exact refer GT at or above this IoU are
# discarded; the original refer GT remains the only target for that instance.
teacher_dedup_iou = 0.5
# Suppress lower-ranked teacher predictions which overlap another retained
# teacher prediction. This prevents duplicate pseudo targets for one object.
teacher_nms_iou = 0.5
if teacher and use_auxiliary_gt:
    raise ValueError("teacher and use_auxiliary_gt are mutually exclusive")
# False: fine-tune all BEiT-3 parameters with lr_vis_enc.
# True: freeze the complete BEiT-3 vision-language backbone.
freeze_beit3 = False

enable_grab=False
# Do not allocate segmentation-only modules; shared 512-channel FPN is unchanged.
enable_mask = False
# Optional observational Top-1 correctness head; never changes boxes/ranking.
# False (or absent in old configs) preserves previous checkpoints/experiments.
enable_correctness = False
correctness = dict(hidden_dim=128, topk=5, duplicate_iou=0.8,
                   loss_weight=1.0, decision_threshold=0.5)

# 聚类自适应Top-N的全图mask CRV；联合训练，无熵/不确定性门控。
# 开启CRV实验时同时设置 enable_birs=False；其他模块开关自行保持。
enable_crv = False
crv_num_groups = 10
crv_grid_size = 48  # Pool complete original/intervened feature maps; no local crops.
crv_start_epoch = 1  # 1-based: no CRV training or reranking before this epoch; 1 starts immediately.


# BIRS: 关闭时旧 CRS、Grab、BCEM 及其权重加载行为保持不变。
# v1：第一版，所有有效候选按外观聚类，簇内原型/中心化，全局重排。
# v2：第二版，CRS Top-N共用原型，只在Top-N内部重排（保留全候选BCE修复）。
# v3：聚类自适应Top-N + 稀疏关系读取 + 显式重绑定对照，仅输出Top-N。
enable_birs = False
birs_version = "v3"  # "v1" / "v2" / "v3"；不改变 enable_bcem 等其他开关。
birs_target_topn = 5
birs_num_groups = 8  # v1/v3使用；不是只保留得分最高的一簇。
birs = dict(version=birs_version, target_topn=birs_target_topn,
            num_groups=birs_num_groups, kmeans_iters=6,  # 仅v1使用。
            reference_topk=64, duplicate_iou=0.8,
            model_dim=128, num_heads=4, chunk_size=32,
            initial_strength=0.1, max_strength=2.0,
            final_loss_weight=1.0, binding_loss_weight=0.1,
            negative_topk=8, detach_inputs=False)
if birs_version == "v3":
    # 不继承v1/v2的额外BCE/绑定排序loss，定位监督总权重保持1。
    birs = dict(version="v3", num_groups=birs_num_groups, kmeans_iters=6,
                seed_topk=3, min_targets=5, max_targets=32,
                reference_topk=64, duplicate_iou=0.8,
                model_dim=128, num_heads=4, chunk_size=8,
                permutations=2, initial_strength=0.1, max_strength=2.0,
                final_mix=0.5, detach_inputs=False, checkpoint_chunks=True)


# CRS -> shared-reference relational refinement. Disabled preserves old models.
# Train alongside CRS; independent of teacher/auxiliary GT; not with CRV.
enable_relation_refinement = False
relation_detach_inputs = True  # True: added final loss trains only stage 2; False: joint gradients.
relation_target_topk = 32
relation_reference_topk = 64  # Detector confidence Top-K FIRST, then duplicate removal.
relation_refinement = dict(
    target_topk=relation_target_topk, reference_topk=relation_reference_topk,
    detach_inputs=relation_detach_inputs,
    duplicate_iou=0.8, model_dim=128, num_heads=4,
    prior_temperature=1.0, relation_temperature=1.0, prior_uniform_mix=0.2,
    discriminability_floor=0.1, initial_strength=0.1, max_strength=2.0,
    loss_weight=1.0, aggregation="competitive",
    use_geometry=True, use_reference_semantics=True,
)

# 背景信息嵌入的候选增强模块
# Candidate background enhancement before CRS; disable for the PropVG baseline.
enable_bcem = False
bcem_sampling_mode = "ray"
bcem_num_rays = 16
# Used only by sector mode; 8 reproduces the original direction order.
bcem_num_sectors = 4

img_size = 586
patch_size = 16
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

train_pipeline = [
    dict(type="LoadAerialVGAnnotations", max_token=max_token,
         use_auxiliary_boxes=use_auxiliary_gt, use_teacher_boxes=teacher),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=True),
    dict(type="Pad", size=(img_size, img_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "text_attention_mask", "gt_bbox"],
         meta_keys=["filename", "expression", "ori_shape", "img_shape", "pad_shape", "scale_factor", "refer_target_index", "empty"]),
]
val_pipeline = [
    dict(type="LoadAerialVGAnnotations", max_token=max_token,
         use_auxiliary_boxes=use_auxiliary_gt, use_teacher_boxes=False),
    dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=True),
    dict(type="Pad", size=(img_size, img_size)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="DefaultFormatBundle"),
    dict(type="CollectData", keys=["img", "ref_expr_inds", "text_attention_mask", "gt_bbox"],
         meta_keys=["filename", "expression", "ori_shape", "img_shape", "pad_shape", "scale_factor", "refer_target_index", "empty"]),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(type=dataset_type, which_set="train", annsfile="data/AerialVG/annotation/vg_train_odvg.jsonl",
               imgsfile="data/AerialVG/images", pipeline=train_pipeline,
               teacher_annsfile=teacher_annsfile if teacher else None,
               teacher_dedup_iou=teacher_dedup_iou,
               teacher_nms_iou=teacher_nms_iou,
               teacher_max_boxes=teacher_max_boxes,
               teacher_max10=teacher_max10),
    val=dict(type=dataset_type, which_set="val", annsfile="data/AerialVG/annotation/vg_val_odvg.jsonl",
             imgsfile="data/AerialVG/images", pipeline=val_pipeline),
    test=dict(type=dataset_type, which_set="test", annsfile="data/AerialVG/annotation/vg_test_odvg.jsonl",
              imgsfile="data/AerialVG/images", pipeline=val_pipeline),
)

model = dict(
    type="MIXRefUniModel_OMG",
    vis_enc=dict(type="BEIT3", img_size=img_size, patch_size=patch_size, vit_type="base", drop_path_rate=0.1,
                 vocab_size=64010, freeze_layer=-1, freeze_beit3=freeze_beit3,
                 vision_embed_proj_interpolate=False,
                 # Load the base BEiT-3 weights only.  This is not a PropVG
                 # task checkpoint; the AerialVG detection/referring head is
                 # still initialized and trained from scratch.
                 pretrain="pretrain_weights/beit3_base_patch16_224.zip"),
    lan_enc=None,
    fusion=None,
    head=dict(type="REFHead", input_channels=768, hidden_channels=256, num_queries=256,
              detr_loss={"criterion": {"loss_class": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
                         "matcher": {"cost_class": 1.0, "cost_bbox": 5.0, "cost_giou": 2.0}},
              loss_weight={"mask": {"dice": 1.0, "bce": 1.0, "nt": 0.2, "neg": 0},
                           "bbox": 0.1, "allbbox": 0.1, "refer": 1.0}, MTD={"K": 250},
              enable_grab=enable_grab, enable_mask=enable_mask,
              enable_correctness=enable_correctness, correctness=correctness,
              enable_relation_refinement=enable_relation_refinement,
              enable_birs=enable_birs, birs=birs,
              relation_refinement=relation_refinement,
              enable_crv=enable_crv,
              crv=dict(num_groups=crv_num_groups, kmeans_iters=6,
                       seed_topk=3, min_targets=3, max_targets=32,
                       grid_size=crv_grid_size,
                       start_epoch=crv_start_epoch,
                       scorer_dim=128, scorer_heads=4, scorer_layers=2, chunk_size=8,
                       inner_scale=1.5, outer_scale=2.0,
                       initial_strength=0.1, max_strength=2.0,
                       negative_gt_coverage=0.01, gt_loss_weight=0.1,
                       cf_loss_weight=0.1, fusion_loss_weight=1.0,
                       positive_margin=1.0, ranking_margin=0.5,
                       checkpoint_chunks=True),
              enable_bcem=enable_bcem,
              bcem=dict(context_dim=128, num_heads=4, num_sectors=bcem_num_sectors,
                        sampling_mode=bcem_sampling_mode, num_rays=bcem_num_rays,
                        radial_edges=(0.0, 0.10, 0.25, 0.50, 1.0),
                        angular_samples=5, radial_samples=4 if bcem_sampling_mode == "ray" else 2,
                        min_valid_points=2 if bcem_sampling_mode == "ray" else 5,
                        context_strength=0.1, chunk_size=32,
                        use_position=True, use_transitions=True, use_gru=True,
                        use_text_attention=True, use_sparse_selection=True,
                        learnable_gate=True, checkpoint_chunks=True)),
    post_params={"score_weighted": False, "mask_threshold": 0.5, "score_threshold": 0.7,
                 "with_nms": False, "with_mask": False},
    process_visual=False,
)

grad_norm_clip = 0.15
use_fp16 = False
ema = False
lr = 0.0004
optimizer_config = dict(type="Adam", lr=lr, lr_vis_enc=0.00004, lr_lan_enc=lr,
                        betas=(0.9, 0.98), eps=1e-9, weight_decay=0, amsgrad=True)
scheduler_config = dict(type="MultiStepLRWarmUp", warmup_epochs=2, decay_steps=[24, 32], decay_ratio=0.1, max_epoch=36)
log_interval = 50
swanlab = dict(mode="online", project="propvg", init_timeout=30, finish_wait=1, stats_interval=30)
start_save_checkpoint = 0
# Keep the last 10 epoch snapshots, plus top1_best.pth and latest.pth.
save_interval = 1
max_keep_ckpts = 10
# finetune_from = "PropVG-grefcoco.pth"
# resume_from = '/media/vision/lzy/PropVG/work_dir/aerialvg/PropVG-bcem-rays-8/20261019_112949/epoch_18.pth'
start_test = 27  # 1-based: test each epoch from 27, using the same GPUs as val; None/<=0 disables.
