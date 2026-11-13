"""Same 30 BASE samples: ROI appearance versus semantic-spatial configuration.

Two environments: propvg Python --stage extract; sklearn Python --stage render.
Only this diagnostic script and a fresh output directory are written.
"""
import argparse
import csv
from functools import partial
import hashlib
import html
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import textwrap
import time
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_appearance_relation_matplotlib")
warnings.filterwarnings("ignore")
import numpy as np

from visualize_decoder_clusters import COLORS, box_iou, l2_features


def load_source(source):
    metadata = json.loads((source / "metadata.json").read_text())
    records = json.loads((source / "samples.json").read_text())
    if len(records) != 30 or [r["index"] for r in records] != metadata["indices"]:
        raise ValueError("Expected the exact previous 30 samples")
    return metadata, records


def select_references(boxes, foreground, topk=64, duplicate_iou=.8):
    valid = np.isfinite(boxes).all(1) & np.isfinite(foreground)
    valid &= (boxes[:, 2:] > boxes[:, :2]).all(1)
    ranked = np.argsort(-np.where(valid, foreground, -np.inf), kind="stable")[:topk]
    kept = []
    for index in ranked:
        if valid[index] and (not kept or np.all(box_iou(boxes[kept], boxes[index]) < duplicate_iou)):
            kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def relation_descriptors(appearance, boxes, foreground, image_size):
    """24 ordered cells, each [confidence-weighted appearance mean, occupancy]."""
    width, height = image_size
    appearance = l2_features(appearance)
    refs = select_references(boxes, foreground)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    offset = (centers[refs][None] - centers[:, None]) / min(width, height)
    distance = np.linalg.norm(offset, axis=-1)
    # Clockwise from up: up / upper-right / right / ... / upper-left.
    angle = np.mod(np.arctan2(offset[..., 0], -offset[..., 1]), 2*np.pi)
    direction = np.floor((angle + np.pi/8) / (np.pi/4)).astype(int) % 8
    radial = np.searchsorted(np.array([.10, .25]), distance, side="right")
    cells = direction*3 + radial
    allowed = np.ones((len(boxes), len(refs)), dtype=bool)
    for index in range(len(boxes)):
        allowed[index] &= refs != index
        allowed[index] &= box_iou(boxes[refs], boxes[index]) < .8
    allowed &= distance > 1e-8  # Undefined direction for coincident centers.
    descriptors = np.zeros((len(boxes), 24, appearance.shape[1]+1), dtype=np.float64)
    counts = np.zeros((len(boxes), 24), dtype=np.int64)
    for index in range(len(boxes)):
        for cell in range(24):
            positions = np.flatnonzero(allowed[index] & (cells[index] == cell))
            if not len(positions):
                continue
            ids = refs[positions]
            weights = np.clip(foreground[ids].astype(float), 1e-12, 1)
            descriptors[index, cell, :-1] = (appearance[ids] * weights[:, None]).sum(0) / weights.sum()
            descriptors[index, cell, -1] = 1.0
            counts[index, cell] = len(ids)
    return descriptors.reshape(len(boxes), -1), refs, allowed, cells, counts


def extract(source, output):
    metadata, records = load_source(source)
    if output.exists():
        raise FileExistsError("Refusing to overwrite: " + str(output))
    experiment = Path(metadata["experiment"])
    sys.path.insert(0, str(experiment))
    import gc
    import torch
    from torchvision.ops import roi_align
    from mmcv import Config
    from mmcv.parallel import collate
    from tqdm import tqdm
    import propvg
    from propvg.datasets import build_dataset, extract_data
    from propvg.datasets.aerialvg_pipeline import LoadAerialVGAnnotations  # noqa: F401
    from propvg.models import build_model

    if Path(propvg.__file__).resolve().parent != experiment / "propvg":
        raise RuntimeError("Must use archived model code")
    config_path = Path(metadata["configuration"])
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != metadata["config_sha256"]:
        raise RuntimeError("Archived config changed")
    torch.set_num_threads(2)
    random.seed(metadata["seed"]); np.random.seed(metadata["seed"]); torch.manual_seed(metadata["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = Config.fromfile(str(config_path))
    cfg.model.vis_enc.pretrain = None
    cfg.model.process_visual = False
    cfg.distributed, cfg.rank, cfg.world_size = False, 0, 1
    for flag in ("enable_grab", "enable_bcem", "enable_crv", "enable_relation_refinement", "enable_birs"):
        if cfg.model.head.get(flag, False):
            raise ValueError("Expected the original BASE, not " + flag)
    dataset = build_dataset(cfg.data[metadata["split"]])
    model = build_model(cfg.model, word_emb=dataset.word_emb, num_token=dataset.num_token)
    print("Strict loading archived BASE: " + metadata["checkpoint"], flush=True)
    checkpoint = torch.load(metadata["checkpoint"], map_location="cpu")
    state = {k[7:] if k.startswith("module.") else k: v for k,v in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    gc.collect()
    model.cuda().eval()
    capture = {}

    def neck_hook(module, inputs, result):
        capture["visual"] = result[0][:, :256]

    original = model.head.forward_test

    def head_forward(*args, **kwargs):
        result = original(*args, **kwargs)
        capture["raw"] = result[0]
        return result

    hook = model.head.neck.register_forward_hook(neck_hook)
    model.head.forward_test = head_forward
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, metadata["indices"]),batch_size=1,
                                        num_workers=0,shuffle=False,collate_fn=partial(collate,samples_per_gpu=1))
    output.mkdir(parents=True)
    (output / "features").mkdir()
    start = time.time()
    fmap_shapes = []
    try:
        with torch.no_grad():
            for record, batch in zip(tqdm(records,desc="ROI appearance: same 30 val samples"), loader):
                meta = batch["img_metas"].data[0][0]
                if str(Path(meta["filename"]).resolve()) != record["filename"] or meta["expression"] != record["expression"]:
                    raise RuntimeError("Sample identity mismatch")
                # No GT boxes/masks enter inference. Archived refer-index
                # metadata remains present but is unused by BASE scoring.
                batch.pop("gt_bbox",None); batch.pop("gt_mask_rle",None); batch.pop("is_crowd",None)
                capture.clear()
                model(**extract_data(batch),return_loss=False,with_bbox=True,with_mask=False,rescale=False)
                raw = capture["raw"]
                normalized = raw["pred_bbox_allobj"][0]
                pixel = torch.cat((normalized[:,:2]-normalized[:,2:]/2, normalized[:,:2]+normalized[:,2:]/2),-1).float()
                h,w = meta["img_shape"][:2]
                ph,pw = meta["pad_shape"][:2]
                pixel *= pixel.new_tensor([w,h,w,h])
                pixel[:,0::2].clamp_(0,w); pixel[:,1::2].clamp_(0,h)
                feature = capture["visual"]
                fh,fw = feature.shape[-2:]
                feature_boxes = pixel * pixel.new_tensor([fw/pw,fh/ph,fw/pw,fh/ph])
                rois = torch.cat((torch.zeros_like(feature_boxes[:,:1]),feature_boxes),-1)
                appearance = roi_align(feature,rois,output_size=(7,7),spatial_scale=1.,sampling_ratio=2,aligned=True)
                appearance = appearance.mean((-2,-1)).float().cpu().numpy()
                scores = raw["refer_label"][0].float().reshape(-1).cpu().numpy()
                foreground = raw["pred_class_allobj"][0].float().softmax(-1)[:,0].cpu().numpy()
                factor = np.asarray(meta["scale_factor"],dtype=float).reshape(-1)
                if factor.size==1: factor=np.repeat(factor,4)
                original_boxes = pixel.cpu().numpy()/factor
                with np.load(source/record["cache"],allow_pickle=False) as cached:
                    np.testing.assert_allclose(original_boxes,cached["boxes_xyxy"],atol=3e-4,rtol=1e-6)
                    np.testing.assert_allclose(foreground,cached["foreground"],atol=1e-6,rtol=1e-5)
                with np.load(source/"scores"/("%04d.npz"%record["index"]),allow_pickle=False) as old:
                    np.testing.assert_allclose(scores,old["crs_scores"],atol=1e-6,rtol=1e-5)
                if appearance.shape != (256,256) or not np.isfinite(appearance).all():
                    raise RuntimeError("Invalid ROI appearance features")
                np.savez_compressed(output/"features"/("%04d.npz"%record["index"]),appearance=appearance,
                                    boxes_xyxy=original_boxes,foreground=foreground,crs_scores=scores,
                                    image_shape=np.array([h,w]),pad_shape=np.array([ph,pw]),feature_shape=np.array([fh,fw]))
                fmap_shapes.append([fh,fw])
    finally:
        hook.remove(); model.head.forward_test = original
    metadata.update(source_cache_directory=str(source),appearance_source="Largest detection-side SimFPN map, channels :256; RoIAlign 7x7 mean",
                    roi_mapping="effective-image pixels multiplied by feature_width/padded_width and feature_height/padded_height",
                    roi_align=dict(aligned=True,sampling_ratio=2,output_size=7),feature_map_shapes=fmap_shapes,
                    physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),extraction_seconds=time.time()-start,
                    peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/2**20,
                    checks=dict(strict_archived_checkpoint=True,same_image_text=True,boxes_match_previous=True,
                                detector_scores_match_previous=True,crs_scores_match_previous=True))
    (output/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    (output/"samples.json").write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print("ROI features saved: "+str(output),flush=True)


def tsne(features, seed):
    from sklearn.manifold import TSNE
    options=dict(n_components=2,perplexity=15,init="pca",learning_rate="auto",random_state=seed,n_jobs=1)
    options["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"]=1500
    return TSNE(**options).fit_transform(l2_features(features))


def cosine_mean(features, indices):
    q=l2_features(features[indices]); similarity=q@q.T
    return float(similarity[np.triu_indices(len(q),1)].mean())


def plot_one(record,cached,labels,a_xy,r_xy,refs,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.patheffects as effects
    from PIL import Image
    boxes=cached["boxes_xyxy"]; scores=cached["crs_scores"]
    ranks=np.argsort(-scores,kind="stable")[:5]
    best=int(np.argmax(cached["ious"]))
    fig,axes=plt.subplots(1,3,figsize=(24,9),gridspec_kw=dict(width_ratios=[1.2,1,1]))
    axes[0].imshow(Image.open(record["filename"]).convert("RGB"))
    refset=set(refs.tolist())
    for query,box in enumerate(boxes):
        x,y,x2,y2=box
        axes[0].add_patch(patches.Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor="#999999",lw=.4,alpha=.20))
        if query in refset:
            axes[0].add_patch(patches.Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor="#00a896",lw=.7,alpha=.55))
    gx,gy,gx2,gy2=record["gt_box"]
    axes[0].add_patch(patches.Rectangle((gx,gy),gx2-gx,gy2-gy,fill=False,edgecolor="white",lw=3,ls="--",
                                      path_effects=[effects.Stroke(linewidth=5,foreground="black"),effects.Normal()]))
    rank_colors=["#f5222d","#ff8c00","#f2d600","#00b9e8","#bc4ae4"]
    legend=[]
    for rank,query in enumerate(ranks):
        x,y,x2,y2=boxes[query]
        axes[0].add_patch(patches.Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor=rank_colors[rank],lw=2))
        axes[0].text(x,y-(rank%2)*7,"T%d q%d"%(rank+1,query),fontsize=8,color="white",
                     bbox=dict(facecolor=rank_colors[rank],alpha=.9,pad=1,edgecolor="none"))
        legend.append("Top-%d q%d: CRS=%.4f, IoU=%.3f"%(rank+1,query,scores[query],cached["ious"][query]))
    bx,by,bx2,by2=boxes[best]
    axes[0].plot((bx+bx2)/2,(by+by2)/2,marker="*",markersize=16,color="white",markeredgecolor="black",markeredgewidth=1)
    axes[0].set_title("Image: 256 boxes; teal = %d references\nDashed white = GT; star = max GT-IoU q%d"%(len(refs),best),fontsize=12)
    axes[0].axis("off")
    for ax,xy,title in zip(axes[1:],(a_xy,r_xy),("ROI appearance t-SNE","Semantic-spatial relation t-SNE")):
        for group in range(8):
            mask=labels==group
            ax.scatter(xy[mask,0],xy[mask,1],s=23,color=COLORS[group],alpha=.72,
                       label="A%d (n=%d)"%(group+1,int(mask.sum())))
        for rank,query in enumerate(ranks):
            ax.scatter(*xy[query],s=170,facecolors="none",edgecolors=rank_colors[rank],linewidths=2.2,zorder=6)
            offsets=[(12,-32),(12,14),(-77,43),(12,67),(-77,-58)]
            ax.annotate("T%d q%d"%(rank+1,query),xy[query],xytext=offsets[rank],textcoords="offset points",fontsize=9,
                        bbox=dict(facecolor="white",alpha=.88,edgecolor=rank_colors[rank],pad=1),
                        arrowprops=dict(arrowstyle="-",color=rank_colors[rank],lw=.8),zorder=8)
        ax.scatter(*xy[best],marker="*",s=260,color="white",edgecolor="black",linewidth=1.5,zorder=7)
        ax.annotate("GT-best q%d"%best,xy[best],xytext=(-95,-9),textcoords="offset points",fontsize=9,
                    bbox=dict(facecolor="white",alpha=.8,pad=1),zorder=9)
        ax.set_title(title+"\nSame 8 appearance-cluster colors; all 256 queries",fontsize=12)
        ax.set_xticks([]);ax.set_yticks([])
        ax.spines[["top","right","bottom","left"]].set_visible(False)
    axes[1].legend(loc="lower left",ncol=2,fontsize=8,framealpha=.75)
    title="BASE | val #%04d | epoch 28\n%s"%(record["index"],"\n".join(textwrap.wrap(record["expression"],155)))
    fig.suptitle(title,fontsize=14,y=.985)
    fig.text(.025,.035,"  |  ".join(legend[:3])+"\n"+"  |  ".join(legend[3:])+
             "\nRelation: detector Top-64, IoU-0.8 dedup; 8 directions x 3 distance bands. Independent t-SNE axes are not comparable.",fontsize=10)
    fig.subplots_adjust(top=.82,bottom=.16,left=.015,right=.99,wspace=.08)
    filename="images/%04d.jpg"%record["index"]
    fig.savefig(output/filename,dpi=160,facecolor="white")
    plt.close(fig)
    return filename


def contact_sheet(output,rows):
    from PIL import Image,ImageDraw
    sheet=Image.new("RGB",(4*900,math.ceil(len(rows)/4)*365),"#edf1f6")
    for index,row in enumerate(rows):
        with Image.open(output/row["visualization"]) as handle:thumb=handle.convert("RGB")
        thumb.thumbnail((892,332),Image.Resampling.LANCZOS)
        x,y=(index%4)*900,(index//4)*365
        sheet.paste(thumb,(x+4,y+26));ImageDraw.Draw(sheet).text((x+8,y+6),"val #%04d"%row["index"],fill="black")
    sheet.save(output/"contact_sheet.jpg",quality=93)


def redraw(output):
    """Layout-only redraw: original high-dimensional vectors/t-SNE stay fixed."""
    from tqdm import tqdm
    _,records=load_source(output)
    rows=[]
    for record in tqdm(records,desc="Improve Top-5 labels (cached t-SNE)"):
        with np.load(output/"descriptors"/("%04d.npz"%record["index"]),allow_pickle=False) as handle:
            cached={k:handle[k].copy() for k in handle.files}
        image=plot_one(record,cached,cached["appearance_clusters"],cached["appearance_tsne"],
                       cached["relation_tsne"],cached["reference_ids"],output)
        rows.append(dict(index=record["index"],visualization=image))
    contact_sheet(output,rows)


def render(source,output):
    from PIL import Image,ImageDraw
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    from tqdm import tqdm
    metadata,records=load_source(output)
    if (output/"index.html").exists():
        raise FileExistsError("Figures already exist; use a fresh output directory")
    (output/"images").mkdir();(output/"descriptors").mkdir()
    rows,query_rows,cards=[],[],[]
    with threadpool_limits(limits=1):
        for record in tqdm(records,desc="Appearance + relation t-SNE (30 images)"):
            with np.load(output/"features"/("%04d.npz"%record["index"]),allow_pickle=False) as handle:
                cached={k:handle[k].copy() for k in handle.files}
            with np.load(source/record["cache"],allow_pickle=False) as old:
                cached["ious"]=old["ious"].copy()
            size=Image.open(record["filename"]).size
            appearance=l2_features(cached["appearance"])
            relation,refs,allowed,cells,counts=relation_descriptors(appearance,cached["boxes_xyxy"],cached["foreground"],size)
            normalized_relation=l2_features(relation)
            labels=KMeans(n_clusters=8,n_init=30,random_state=metadata["seed"],algorithm="lloyd").fit_predict(appearance)
            a_xy,r_xy=tsne(appearance,metadata["seed"]),tsne(normalized_relation,metadata["seed"])
            top5=np.argsort(-cached["crs_scores"],kind="stable")[:5]
            image=plot_one(record,cached,labels,a_xy,r_xy,refs,output)
            np.savez_compressed(output/"descriptors"/("%04d.npz"%record["index"]),appearance=appearance,
                                relation=relation,normalized_relation=normalized_relation,
                                appearance_tsne=a_xy,relation_tsne=r_xy,appearance_clusters=labels,
                                reference_ids=refs,reference_allowed=allowed,reference_cells=cells,cell_counts=counts,
                                crs_scores=cached["crs_scores"],ious=cached["ious"],boxes_xyxy=cached["boxes_xyxy"])
            row=dict(index=record["index"],filename=record["filename"],reference_count=len(refs),
                     top5_appearance_cosine=cosine_mean(appearance,top5),top5_relation_cosine=cosine_mean(relation,top5),
                     top1_query=int(top5[0]),top1_iou=float(cached["ious"][top5[0]]),
                     max_iou_query=int(np.argmax(cached["ious"])),max_iou=float(cached["ious"].max()),
                     mean_occupied_cells=float((counts>0).sum(1).mean()),zero_relation_queries=int((np.linalg.norm(relation,axis=1)==0).sum()),
                     visualization=image)
            rows.append(row)
            for query in range(256):
                match=np.flatnonzero(top5==query)
                query_rows.append(dict(sample=record["index"],query=query,appearance_cluster=int(labels[query])+1,
                                       crs_score=float(cached["crs_scores"][query]),crs_top5_rank=int(match[0])+1 if len(match) else 0,
                                       gt_iou=float(cached["ious"][query]),legal_references=int(allowed[query].sum()),
                                       occupied_cells=int((counts[query]>0).sum()),appearance_x=float(a_xy[query,0]),
                                       appearance_y=float(a_xy[query,1]),relation_x=float(r_xy[query,0]),relation_y=float(r_xy[query,1])))
            cards.append('<article><h2>val #%04d | references=%d</h2><p>%s</p><a href="%s"><img loading="lazy" src="%s"></a></article>'%
                         (record["index"],len(refs),html.escape(record["expression"]),html.escape(image),html.escape(image)))
    for filename,data in (("per_sample.csv",rows),("per_query.csv",query_rows)):
        with (output/filename).open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    summary=dict(samples=30,queries_per_sample=256,source=str(source),experiment=metadata["experiment"],
                 checkpoint=metadata["checkpoint"],indices=metadata["indices"],seed=metadata["seed"],
                 appearance_dimension=256,relation_dimension=24*257,
                 relation_reference_topk=64,duplicate_iou=.8,distance_edges_short_side=[.1,.25],
                 angular_order=["up","upper-right","right","lower-right","down","lower-left","left","upper-left"],
                 tsne=dict(perplexity=15,iterations=1500,init="pca",input_normalization="L2"),
                 mean_top5_appearance_cosine=float(np.mean([r["top5_appearance_cosine"] for r in rows])),
                 mean_top5_relation_cosine=float(np.mean([r["top5_relation_cosine"] for r in rows])),
                 reference_count_min=min(r["reference_count"] for r in rows),reference_count_max=max(r["reference_count"] for r in rows),
                 zero_relation_queries=sum(r["zero_relation_queries"] for r in rows),
                 caveat="ROI features inherit backbone context/text. Handcrafted relation descriptor is not a learned relation score; cosine scales differ.")
    (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Appearance vs relation t-SNE</title>'
        '<style>body{font:18px sans-serif;max-width:2400px;margin:24px auto;background:#edf1f6}article{background:white;padding:16px;margin:24px 0}img{width:100%}</style>'
        '<h1>30 张验证图：框内外观与语义—空间配置 t-SNE</h1>'
        '<p>全部256个候选保留。颜色来自框内外观的8类聚类，左右t-SNE共用颜色；没有重新按关系着色。'
        'Top-1～5按原CRS得分标注；星号为与GT最大IoU的query。两张独立t-SNE坐标不能直接比较。'
        '<a href="README.md">方法与解释边界</a> | <a href="per_sample.csv">相似度统计</a></p>'+''.join(cards))
    contact_sheet(output,rows)
    report="""# 同一组30张图：框内外观与语义—空间配置

## 来源

沿用 BASE `20261105_105035`、epoch 28 `top1_best.pth`，归档配置及源码严格加载。
样本与之前的30张完全相同；新前向的框、检测置信度、CRS得分均与旧缓存逐query校验。
没有改动生产模型、配置、权重、旧图或旧缓存。只重新前向这30个样本，不训练、不跑完整验证集。

## 外观描述

取原 SimFPN 最高分辨率检测侧特征（前256通道），对256个预测框分别 RoIAlign 7×7
（aligned=True, sampling_ratio=2）再空间平均，得到 `[256,256]`。
坐标先按有效图像裁剪，再用 `feature_size / pad_size` 映射到特征图，考虑padding。
对每个候选L2归一化。该表示更集中在框内，但深层特征仍有感受野上下文且来自多模态骨干，
不能称为完全纯净、与文本无关的外观。

## 关系描述

只把检测前景置信度 Top-64、IoU≥0.8 去重后的框作为参考池，不补齐、不使用GT或CRS分数选参考。
对每个目标候选，排除自身及IoU≥0.8的近重复框；部分重叠允许。
中心重合、方向无定义的参考不进入配置统计。

按目标中心的相对方向划8格，从正上方起顺时针；距离为中心距/原图短边，分为
`[0,0.1)`、`[0.1,0.25)`、`[0.25,+inf)`，共24格。每格对参考物的L2外观特征按检测置信度
加权平均，拼接一个0/1占用标记（权重1），空格全零。依固定方向/距离顺序拼接，
得到 `[256,6168]`，整体L2归一化后做t-SNE。

这是人为定义的语义—几何配置描述，不是学到的关系评分，也不含目标自身外观或显式文本/GT输入。
它依赖参考物召回，不能覆盖未被候选表示的道路、建筑等所有参照；关系构造没有新增文本处理，
但参考外观继承了上游模型的多模态信息。固定距离阈值只是诊断设定，不是验证集调出的最优超参数。

## 展示与统计

- 全部256点保留；参考Top-64仅影响关系支路，不裁掉待分析的query。
- 左图灰色细框为全部候选，青绿色为去重参考池；白色虚线为GT；彩色粗框为CRS Top-1～5。
- 中右图使用相同的外观K-Means8类颜色，K-Means在256维外观上完成，不在t-SNE坐标上聚类。
- 星号为GT最大IoU的query，GT只用于离线标注/统计，不参与特征构造或聚类。
- 两组t-SNE均固定种子、perplexity=15、PCA初始化、1500迭代；轴、方向、尺度不跨图对齐。
- `per_sample.csv` 给出Top-5在原始高维描述中的两两余弦均值。
  外观和关系描述结构、维度、占用信号不同，不能仅因关系余弦更低就断言定位更有效；
  需看能否把GT对应候选从错误竞争者中区分，以及不同空间划分设置下是否稳定。
- `descriptors/` 保存两套高维特征、t-SNE、参考下标、逐候选合法参考及空间格、query对齐的得分和IoU。

输出：`index.html` 图库，`images/` 30张完整图，`contact_sheet.jpg` 总览，
`per_sample.csv` / `per_query.csv` 统计，`metadata.json` / `summary.json` 来源和设置。
"""
    (output/"README.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("extract","render","redraw"),required=True)
    parser.add_argument("--source",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    source,output=Path(args.source).resolve(),Path(args.output).resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate sibling output directory")
    if args.stage=="extract":extract(source,output)
    elif args.stage=="render":render(source,output)
    else:redraw(output)
