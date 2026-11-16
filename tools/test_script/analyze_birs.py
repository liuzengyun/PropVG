"""Read-only archived-checkpoint CRS vs BIRS audit; one forward per sample."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
warnings.filterwarnings("ignore")
import numpy as np


def metrics(records, mode):
    return {name: float(100*np.mean([row[mode+"_"+key] for row in records]))
            for name,key in (("top1","top1_correct"),("top5","top5_correct"),("miou","top1_iou"))}


def write_report(output, records, provenance):
    old, new = metrics(records,"crs"), metrics(records,"final")
    corrected=sum(row["corrected"] for row in records)
    harmed=sum(row["harmed"] for row in records)
    changed=sum(row["rank_changed"] for row in records)
    assert abs(new["top1"]-old["top1"]-100*(corrected-harmed)/len(records))<1e-8
    result=dict(**provenance,samples=len(records),crs=old,final=new,corrected=corrected,harmed=harmed,
                net=corrected-harmed,net_top1_points=100*(corrected-harmed)/len(records),rank_changed=changed,
                both_correct_changed=sum(r["rank_changed"] and r["crs_top1_correct"] and r["final_top1_correct"] for r in records),
                both_wrong_changed=sum(r["rank_changed"] and not r["crs_top1_correct"] and not r["final_top1_correct"] for r in records),
                mean_abs_delta=float(np.mean([r["delta_abs_mean"] for r in records])),
                eligible_fraction=float(np.mean([r["eligible_fraction"] for r in records])),
                mean_raw_binding_std=float(np.mean([r["raw_binding_std"] for r in records])))
    with (output/"per_sample.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for name,flag in (("corrected.csv","corrected"),("harmed.csv","harmed")):
        with (output/name).open("w",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader()
            writer.writerows(row for row in records if row[flag])
    (output/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=["# BIRS 同模型原 CRS / 最终排序诊断", "",
           "样本 {}；split={}；epoch={}。".format(len(records),provenance["split"],provenance["checkpoint_epoch"]),"",
           "| 排序 | Top-1 | Top-5 | box mIoU |","|---|---:|---:|---:|"]
    for label,values in (("原 CRS（同一联合训练模型）",old),("CRS+BIRS",new)):
        lines.append("| {} | {:.4f}% | {:.4f}% | {:.4f}% |".format(label,values["top1"],values["top5"],values["miou"]))
    lines += ["", "改对 {}，改错 {}，净变化 {}；Top-1 变化 {:+.4f} 个百分点。".format(corrected,harmed,corrected-harmed,result["net_top1_points"]),
              "Top-1 query 改变 {} 个，其中改前改后都正确 {} 个，都错误 {} 个。".format(changed,result["both_correct_changed"],result["both_wrong_changed"]),"",
              "## 含义与限制","",
              "原 CRS 是此 checkpoint 联合训练后、加 BIRS 残差之前的得分，不是独立训练的旧 BASE。",
              "该实验 detach_inputs=False，新增 final BCE 和 binding loss 已参与特征/CRS 学习；关闭推理残差不能撤销训练影响。",
              "因此净纠正量衡量的是推理时残差的直接贡献；要判断整个训练方案，需要同配置独立 baseline 及额外定位损失控制实验。",
              "GT 只用于本脚本离线 IoU，不进入模型前向。框及分数来源与归档权重严格匹配。",
              "默认按原概率 CPU argsort 得到 CRS 排名，按原后处理 ranking_logits 的 stable argsort 得到最终排名。",
              "所有样本的最终 Top-1 框、分数及全候选顺序都与正常模型输出交叉校验；IoU 阈值0.5，mIoU取Top-1。", "",
              "原始逐 query 证据保存在 details/*.npz，逐样本和改对/改错清单保存在 CSV。", "",
              "```json",json.dumps(provenance,ensure_ascii=False,indent=2),"```"]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--checkpoint",default="epoch_34.pth")
    parser.add_argument("--splits",nargs="+",choices=("val","test"),default=["val","test"])
    parser.add_argument("--output",required=True)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    experiment,output=Path(args.experiment).resolve(),Path(args.output).resolve()
    if output.exists():raise FileExistsError("Refusing to overwrite existing audit: "+str(output))
    # Archived model import precedes any training framework imports.
    sys.path.insert(0,str(experiment))
    import torch
    import propvg
    from mmcv import Config
    from tqdm import tqdm
    from propvg.datasets import build_dataset,build_dataloader,extract_data
    from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
    from propvg.models import build_model
    from propvg.utils.checkpoint import _validate_birs_checkpoint
    if Path(propvg.__file__).resolve().parent != experiment/"propvg":raise RuntimeError("Not archived code")
    paths=list(experiment.glob("*_PropVG-aerialvg.py"))
    if len(paths)!=1:raise RuntimeError("Expected one saved config")
    config_path=paths[0];cfg=Config.fromfile(str(config_path))
    if not cfg.model.head.get("enable_birs",False):raise ValueError("BIRS must be enabled in saved config")
    cfg.model.vis_enc.pretrain=None
    cfg.model.process_visual=False
    cfg.distributed,cfg.rank,cfg.world_size=False,0,1
    cfg.data.samples_per_gpu,cfg.data.workers_per_gpu=args.batch_size,args.workers
    torch.set_num_threads(2);torch.manual_seed(cfg.seed or 0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    first=build_dataset(cfg.data[args.splits[0]])
    model=build_model(cfg.model,word_emb=first.word_emb,num_token=first.num_token)
    path=experiment/args.checkpoint
    print("Strictly loading "+str(path),flush=True)
    checkpoint=torch.load(path,map_location="cpu")
    state={key[7:] if key.startswith("module.") else key:value for key,value in checkpoint["state_dict"].items()}
    _validate_birs_checkpoint(model,state)
    model.load_state_dict(state,strict=True)
    epoch=int(checkpoint.get("epoch",-1))+1
    del state,checkpoint;gc.collect()
    model.cuda().eval()
    stage=model.head.birs
    if int(stage.trained_steps)<=0:raise RuntimeError("No supervised BIRS steps")
    provenance=dict(experiment=str(experiment),checkpoint=str(path),checkpoint_epoch=epoch,
                    checkpoint_bytes=path.stat().st_size,checkpoint_mtime_ns=path.stat().st_mtime_ns,
                    imported_source=propvg.__file__,config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    trained_steps=int(stage.trained_steps),strength=float(stage.max_strength*stage.strength_logit.sigmoid()),
                    batch_size=args.batch_size,physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    configuration=dict(teacher=cfg.get("teacher"),enable_grab=cfg.get("enable_grab"),enable_bcem=cfg.get("enable_bcem"),
                                       img_size=cfg.get("img_size"),birs=dict(cfg.model.head.birs)))
    captured={}
    wanted=("boxes","valid","eligible","groups","base_scores","final_logits","raw_binding","paired","null","delta",
            "reference_indices","reference_valid","ranking_priority")

    def hook(module,inputs,result):
        captured.clear()
        captured.update({key:result[key].detach().cpu() for key in wanted})
        # V3 keeps full logits for BCE but exports only its adaptive prefix.
        for key in ("output_mask", "candidate_mask", "target_indices", "target_valid"):
            if key in result:
                captured[key] = result[key].detach().cpu()

    handle=stage.register_forward_hook(hook)
    output.mkdir(parents=True)
    all_summaries={}
    try:
        with torch.no_grad():
            for split in args.splits:
                dataset=first if split==args.splits[0] else build_dataset(cfg.data[split])
                loader=build_dataloader(cfg,dataset)
                folder=output/split;folder.mkdir();(folder/"details").mkdir()
                records=[];started=time.time()
                for inputs in tqdm(loader,desc="BIRS audit "+split,dynamic_ncols=True):
                    metas=inputs["img_metas"].data[0]
                    target_boxes=inputs["gt_bbox"].data[0]
                    targets=[]
                    for boxes,meta in zip(target_boxes,metas):
                        indices=meta["refer_target_index"]
                        if len(indices)!=1:raise ValueError("Expected exactly one refer GT")
                        targets.append(boxes.reshape(-1,4)[indices[0]].float().cpu())
                    for name in ("gt_bbox","gt_mask_rle","is_crowd"):inputs.pop(name,None)
                    predictions=model(**extract_data(inputs),return_loss=False,with_bbox=True,with_mask=False,rescale=False)["pred_bboxes"]
                    for bi,(meta,target,pred) in enumerate(zip(metas,targets,predictions)):
                        index=len(records);sample=dataset.samples[index]
                        if Path(sample["filename"]).name!=Path(meta["filename"]).name:raise RuntimeError("Dataset order mismatch")
                        detail={key:value[bi] for key,value in captured.items()}
                        valid=detail["valid"]
                        ids=valid.nonzero(as_tuple=True)[0]
                        before=ids[detail["base_scores"][ids].argsort(descending=True)]
                        final_valid=valid & detail.get("output_mask",valid)
                        final_ids=final_valid.nonzero(as_tuple=True)[0]
                        after=final_ids[detail["ranking_priority"].reshape(-1)[final_ids].argsort(descending=True,stable=True)]
                        h,w=meta["img_shape"][:2]
                        boxes=detail["boxes"]*torch.tensor([w,h,w,h])
                        intersection=(torch.minimum(boxes[:,2:],target[2:])-torch.maximum(boxes[:,:2],target[:2])).clamp(min=0).prod(1)
                        iou=intersection/((boxes[:,2:]-boxes[:,:2]).clamp(min=0).prod(1)+(target[2:]-target[:2]).clamp(min=0).prod()-intersection).clamp(min=1e-6)
                        actual=pred["ranking_logits"].reshape(-1).cpu().argsort(descending=True,stable=True)
                        torch.testing.assert_close(pred["boxes"][actual].float().cpu(),boxes[after],rtol=1e-5,atol=2e-3)
                        torch.testing.assert_close(pred["scores"].reshape(-1).cpu(),detail["final_logits"].reshape(-1)[final_valid].sigmoid(),rtol=1e-5,atol=1e-7)
                        row=dict(dataset_index=index,filename=meta["filename"],expression=sample["expression"],
                                 eligible_fraction=float(detail["eligible"].float().mean()),delta_abs_mean=float(detail["delta"].abs().mean()),
                                 raw_binding_std=float(detail["raw_binding"].std()),reference_count=int(detail["reference_valid"].sum()))
                        for mode,order in (("crs",before),("final",after)):
                            row.update({mode+"_top1_query":int(order[0]) if len(order) else -1,
                                        mode+"_top1_iou":float(iou[order[0]]) if len(order) else 0.,
                                        mode+"_top1_correct":bool(len(order)) and bool(iou[order[0]]>=.5),
                                        mode+"_top5_correct":bool(len(order)) and bool(iou[order[:5]].max()>=.5)})
                        row.update(corrected=not row["crs_top1_correct"] and row["final_top1_correct"],
                                   harmed=row["crs_top1_correct"] and not row["final_top1_correct"],
                                   rank_changed=row["crs_top1_query"]!=row["final_top1_query"])
                        records.append(row)
                        arrays={key:value.numpy() for key,value in detail.items()}
                        arrays.update(gt_pixel_xyxy=target.numpy(),ious=iou.numpy(),crs_order=before.numpy(),final_order=after.numpy())
                        np.savez_compressed(folder/"details"/("%06d.npz"%index),**arrays)
                if len(records)!=len(dataset):raise RuntimeError("Incomplete dataset")
                all_summaries[split]=write_report(folder,records,dict(**provenance,split=split,elapsed_seconds=time.time()-started))
                (output/"summary.json").write_text(json.dumps(all_summaries,ensure_ascii=False,indent=2))
    finally:handle.remove()


if __name__=="__main__":main()
