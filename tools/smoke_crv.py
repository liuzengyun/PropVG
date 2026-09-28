"""Real-data independent CRV check; no training run, optimizer or checkpoint."""

import argparse
import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")

import torch
from mmcv import Config
from propvg.datasets import build_dataset
from propvg.models import build_model


def pack_sample(sample, device):
    return dict(
        img=sample["img"].data[None].to(device),
        ref_expr_inds=sample["ref_expr_inds"].data[None].to(device),
        text_attention_mask=sample["text_attention_mask"].data[None].to(device),
        img_metas=[sample["img_metas"].data],
        gt_bbox=[sample["gt_bbox"].data.to(device)],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=.15)
    parser.add_argument("--report", required=True)
    parser.add_argument("--enable-crv", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(6666)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cfg = Config.fromfile(args.config)
    if args.enable_crv:
        cfg.model.head.enable_crv = True
    train_dataset, val_dataset = build_dataset(cfg.data.train), build_dataset(cfg.data.val)
    train_inputs, val_inputs = pack_sample(train_dataset[0], device), pack_sample(val_dataset[0], device)
    model = build_model(cfg.model, word_emb=train_dataset.word_emb, num_token=train_dataset.num_token).to(device)
    if not model.head.enable_crv:
        raise ValueError("Enable CRV in the diagnostic config or pass --enable-crv")
    captured = {}

    def capture(_module, _args, kwargs, output):
        captured.update(output=output, kwargs=kwargs)

    hook = model.head.crv.register_forward_hook(capture, with_kwargs=True)
    model.eval()
    with torch.no_grad():
        image_tokens, language, cls = model.extract_visual_language(
            train_inputs["img"], train_inputs["ref_expr_inds"], train_inputs["text_attention_mask"])
    b, _, h, w = train_inputs["img"].shape
    spatial = image_tokens.transpose(-1, -2).reshape(b, -1, h // model.patch_size, w // model.patch_size)
    model.head.train()
    target = dict(mask=None, bbox=train_inputs["gt_bbox"], img_metas=train_inputs["img_metas"])
    losses, _, _ = model.head.forward_train(spatial, target, cls_feat=cls, lan_feat=language,
                                          lan_mask=train_inputs["text_attention_mask"], img=train_inputs["img"])
    assert all(torch.isfinite(v) for v in losses.values())
    losses["loss_crv"].backward(retain_graph=True)
    assert all(p.grad is None for name, p in model.named_parameters() if not name.startswith("head.crv."))
    assert all(p.grad is not None for p in model.head.crv.parameters())
    model.zero_grad(set_to_none=True)
    sum(value for key, value in losses.items() if key.startswith("loss_")).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.mrsm.parameters())
    assert all(p.grad is None for p in model.vis_enc.parameters())
    train_capture, train_arguments = captured["output"], captured["kwargs"]
    expected_gt = model.head.prepare_targets(train_inputs["gt_bbox"], train_inputs["img_metas"])[0][0]["boxes"][0]
    torch.testing.assert_close(train_arguments["target_boxes"][0], expected_gt)
    model.zero_grad(set_to_none=True)
    captured.clear()
    model.eval()
    with torch.no_grad():
        prediction = model(**{k: v for k, v in val_inputs.items() if k != "gt_bbox"},
                           return_loss=False, with_bbox=True, with_mask=False, rescale=False)
    assert "target_boxes" not in captured["kwargs"]
    assert captured["output"]["gt_support_drop_logits"] is None
    assert prediction["pred_bboxes"][0]["boxes"].shape == (8, 4)
    hook.remove()
    report = dict(
        config=args.config, elapsed_seconds=time.perf_counter() - started,
        scope="pretrained BEiT-3 no_grad; head backward; val inference; no optimizer/checkpoint",
        train_size=len(train_dataset), val_size=len(val_dataset),
        train_filename=train_inputs["img_metas"][0]["filename"],
        image_shape=list(train_inputs["img"].shape), text_shape=list(train_inputs["ref_expr_inds"].shape),
        crv_visual_shape=list(train_arguments["visual"].shape),
        crv_text_shape=list(train_arguments["text"].shape),
        valid_ratios=train_arguments["valid_ratios"].detach().cpu().tolist(),
        detection_gt_count=len(train_inputs["gt_bbox"][0]),
        refer_gt_normalized=train_arguments["target_boxes"].detach().cpu().tolist(),
        candidate_count=int(train_arguments["boxes"].shape[1]), topk=model.head.crv.topk,
        crv_parameter_count=sum(p.numel() for p in model.head.crv.parameters()),
        losses={k: float(v) for k, v in losses.items()},
        positive_candidates=int(train_capture["positive_candidates"].sum()),
        safe_negative_candidates=int(train_capture["negative_candidates"].sum()),
        crv_gradient_isolation=True, baseline_crs_gradients=True,
        inference_has_no_gt=True, inference_boxes_shape=list(prediction["pred_bboxes"][0]["boxes"].shape),
        peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024 ** 2,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device) / 1024 ** 2,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
