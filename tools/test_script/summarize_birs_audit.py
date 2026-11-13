"""Summarize completed BIRS audit caches without further inference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest


def summarize_folder(folder):
    result=json.loads((folder/"summary.json").read_text())
    margins=[]; opportunities=[]; same_group=[]
    for path in sorted((folder/"details").glob("*.npz")):
        with np.load(path,allow_pickle=False) as z:
            order=z["crs_order"]; iou=z["ious"]; delta=z["delta"]
            logits=z["final_logits"].reshape(-1)-delta
            if len(order)>1:margins.append(float(logits[order[0]]-logits[order[1]]))
            good=order[:5][iou[order[:5]]>=.5]
            if len(good) and iou[order[0]]<.5:
                target=int(good[0]);wrong=int(order[0])
                advantage=float(delta[target]-delta[wrong])
                gap=float(logits[wrong]-logits[target])
                opportunities.append((advantage,gap))
                same_group.append(bool(z["groups"][target]==z["groups"][wrong]))
    values=np.asarray(opportunities).reshape(-1,2)
    result["further_diagnostics"]=dict(
        median_crs_top1_top2_logit_margin=float(np.median(margins)),
        mean_crs_top1_top2_logit_margin=float(np.mean(margins)),
        crs_wrong_top5_reachable=len(values),
        reachable_same_feature_group=int(sum(same_group)),
        reachable_delta_towards_gt=int((values[:,0]>0).sum()),
        reachable_delta_away_from_gt=int((values[:,0]<0).sum()),
        reachable_delta_exceeds_original_gap=int((values[:,0]>values[:,1]).sum()),
        reachable_mean_delta_advantage=float(values[:,0].mean()),
        reachable_median_original_gap=float(np.median(values[:,1])),
        paired_mcnemar_exact_p=float(binomtest(result["corrected"],result["corrected"]+result["harmed"],.5).pvalue)
            if result["corrected"]+result["harmed"] else 1.)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val",required=True);parser.add_argument("--test",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result={split:summarize_folder(Path(getattr(args,split))) for split in ("val","test")}
    output=Path(args.output)
    (output/"combined_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines=["# epoch_34 BIRS 完整诊断","", "同一 checkpoint、同一前向的原 CRS 与最终 BIRS 排序对照。", "",
           "| 数据集 | 模式 | Top-1 | Top-5 | box mIoU |","|---|---|---:|---:|---:|"]
    for split,summary in result.items():
        for mode in ("crs","final"):
            m=summary[mode]
            lines.append("| {} | {} | {:.4f}% | {:.4f}% | {:.4f}% |".format(split,mode,m["top1"],m["top5"],m["miou"]))
    lines += ["", "| 数据集 | 改对 | 改错 | 净改对 | Top-1变化/百分点 | Top-1 query改变 |",
              "|---|---:|---:|---:|---:|---:|"]
    for split,s in result.items():
        lines.append("| {} | {} | {} | {:+d} | {:+.4f} | {} |".format(split,s["corrected"],s["harmed"],s["net"],s["net_top1_points"],s["rank_changed"]))
    lines += ["", "## 解释", "",
              "这里的原 CRS 来自联合训练后的 epoch_34，不是独立 BASE。`detach_inputs=False`，训练影响无法通过测试时移除残差撤销。",
              "原 BASE `20261105_105035` 的最佳结果为 val 67.06/88.19/58.32、test 66.95/88.44/58.31。",
              "已核对两次归档配置的 teacher、图像尺寸586、原loss权重、随机种子、数据配置、优化器、学习率调度一致；",
              "但 epoch 与额外损失不同，且 BIRS 训练日志中的最终融合BCE是额外一次定位监督。",
              "不能把相对旧BASE的所有提升归因于推理时的关系残差；需要定位损失重加权对照及绑定损失消融。", "",
              "## 进一步诊断（原CRS错误、但正确候选在Top-5）", ""]
    for split,s in result.items():
        d=s["further_diagnostics"]
        lines += ["### "+split,"", "样本数 {}；正确候选与错误Top-1同特征组 {}。".format(d["crs_wrong_top5_reachable"],d["reachable_same_feature_group"]),
                  "残差朝正确候选方向推动 {} 个，反方向 {} 个，超过原CRS分差 {} 个。".format(d["reachable_delta_towards_gt"],d["reachable_delta_away_from_gt"],d["reachable_delta_exceeds_original_gap"]),
                  "这些样本原CRS分差中位数 {:.4f}，残差优势均值 {:.4f}。".format(d["reachable_median_original_gap"],d["reachable_mean_delta_advantage"]),
                  "全部样本Top-1/Top-2原logit分差中位数 {:.4f}；可学习强度β={:.4f}，平均|δ|={:.4f}。".format(d["median_crs_top1_top2_logit_margin"],s["strength"],s["mean_abs_delta"]),
                  "改对/改错精确配对检验 p={:.4g}（仅衡量该checkpoint推理残差的方向差异，不是跨训练种子的显著性）。".format(d["paired_mcnemar_exact_p"]), ""]
    lines += ["## 原始记录", "",str(Path(args.val).resolve()/"per_sample.csv"),str(Path(args.test).resolve()/"per_sample.csv"),"",
              "测试均单卡、逐样本不重复。训练时多卡评估可能将尾部补齐到world_size整倍数，故日志四舍五入值可能相差0.01个百分点。",
              "本次没有修改训练代码或模型，不进行阈值/权重调优。"]
    (output/"analysis_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
