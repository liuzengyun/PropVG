# PropVG 在 AerialVG 上的完整数据流

本文档记录当前 `configs/aerialvg/PropVG-aerialvg.py` 配置下，AerialVG 数据从加载、预处理、BEiT-3 编码、候选框生成、文本匹配，到最终输出目标框的完整流程及张量维度。

## 1. 当前配置

```python
teacher = True
use_auxiliary_gt = False
enable_grab = False
enable_crv = True
crv_topk = 8
freeze_beit3 = False

img_size = 586
patch_size = 16
max_token = 64
num_queries = 256
samples_per_gpu = 8
```

因此当前实验是 **PropVG + teacher proposal GT + CRV**，Grab 当前关闭。CRV 的完整设计、损失和维度见项目根目录的 `CRV_DESIGN.md`。

下面使用以下符号：

- `B`：每张 GPU 的 batch size，当前为 8。
- `Ni`：第 `i` 张图的检测监督框数量，当前为原始 refer GT 加 NMS 后的 teacher 框。
- 使用 GPU 2、3 进行双卡训练时，每卡仍为 `B=8`，全局 batch size 为 16。

## 2. 总体流程

```text
AerialVG JSONL + teacher_trainset.json
                ↓
读取图像、完整文本、原始 refer GT、teacher 候选框
                ↓
teacher/refer 去重 + teacher 内部 NMS
                ↓
文本分词到 64 token；图像等比例缩放并 pad 到 586×586
                ↓
BEiT-3 联合编码图像与文本
                ↓
图像特征 [B,768,36,36]；文本特征 [B,64,768]
                ↓
通道投影 + SimFPN 多尺度特征
                ↓
Deformable DETR Decoder + 256 个 query
                ↓
生成 256 个候选框及候选特征
                ↓
CRS 计算每个候选框与文本的相关性
                ↓
CRV 取原始 CRS Top-8，执行反事实验证并融合重排
                ↓
最高分候选框作为最终 Top-1 定位结果
```

## 3. 数据集标注加载

### 3.1 原始 AerialVG 标注

训练集来自：

```text
data/AerialVG/annotation/vg_train_odvg.jsonl
```

每一行是一张图和一段 referring expression：

```json
{
  "filename": "example.jpg",
  "width": 1400,
  "height": 788,
  "grounding": {
    "caption": "A white truck was driving in the street...",
    "regions": [
      {"bbox": [x1, y1, x2, y2], "phrase": "A white truck"}
    ]
  }
}
```

代码中的约定是：

```text
regions[0]  = 需要定位的原始 refer GT
regions[1:] = AerialVG 自带的其他区域标注
```

当前 `use_auxiliary_gt=False`，因此 `regions[1:]` 不参与训练。

### 3.2 Teacher 标注

teacher 框来自：

```text
data/AerialVG/teacher_trainset.json
```

处理顺序：

1. teacher 框裁剪到原图范围。
2. 删除宽度或高度无效的框。
3. 删除与原始 refer GT 的 IoU 大于等于 `teacher_dedup_iou=0.5` 的框。
4. teacher 框之间执行 `teacher_nms_iou=0.5` 的贪心 NMS，保留排名更高的框。
5. 组成该样本的检测训练目标。

```text
gt_bbox = [原始 refer GT] + [保留下来的 teacher boxes]
gt_bbox shape = [Ni,4]
```

其中：

```text
1 <= Ni <= 6
gt_bbox[0] 永远是原始 refer GT
refer_target_index = [0]
```

teacher 框只提供 proposal/前景检测监督，不会替代原始 refer GT。

验证集和测试集不加载 teacher，而且当前 `use_auxiliary_gt=False`，所以：

```text
val/test gt_bbox shape = [1,4]
```

## 4. 图像和文本预处理

### 4.1 图像

原图首先按照最长边 586 等比例缩放，然后 pad 到正方形：

```python
Resize(img_scale=(586, 586), keep_ratio=True)
Pad(size=(586, 586))
Normalize(...)
```

例如原图为 `1400×788` 时：

```text
原图：      [788,1400,3]
resize 后： [330,586,3]（近似）
pad 后：    [586,586,3]
网络输入：  [B,3,586,586]
```

padding 位于有效图像的右侧和下侧。

原始 GT 与 teacher 框会同步乘 resize scale，始终保持像素 `xyxy` 格式：

```text
[x1,y1,x2,y2]
```

此时坐标对应 resize 后的有效图像，不包含 padding 坐标偏移。

### 4.2 文本

文本处理顺序：

1. 转为小写。
2. 清理部分标点。
3. 将 `-` 和 `/` 替换为空格。
4. 使用 BEiT-3 SentencePiece tokenizer。
5. 添加 BOS 和 EOS。
6. 截断或 padding 到 64 token。

输出：

```text
ref_expr_inds:       [B,64]，int64
text_attention_mask: [B,64]，int64
```

mask 定义：

```text
0 = 有效 token
1 = padding token
```

64 个位置中最多包含 62 个正文 token，其余两个位置用于 BOS/EOS。

## 5. BEiT-3 联合编码

BEiT-3 Base 配置：

```text
Transformer layers = 12
hidden dimension   = 768
attention heads    = 12
patch size         = 16
```

输入：

```text
image: [B,3,586,586]
text:  [B,64]
mask:  [B,64]
```

图像 patch 网格：

```text
586 // 16 = 36
36 × 36 = 1296 visual tokens
```

联合序列大致为：

```text
[CLS] + 1296 visual tokens + 64 text tokens
总长度 = 1 + 1296 + 64 = 1361
```

BEiT-3 输出：

```text
encoder_out: [B,1361,768]
```

拆分为：

```text
cls_feat:  [B,768]
img_feat:  [B,1296,768]
text_feat: [B,64,768]
```

图像 token reshape 为二维特征图：

```text
[B,1296,768]
→ [B,768,1296]
→ [B,768,36,36]
```

当前 `freeze_beit3=False`，所以 BEiT-3 参与微调，学习率为 `lr_vis_enc=4e-5`。

注意：586 不能被 patch size 16 整除。卷积 patch embedding 实际覆盖 `576×576`，最右侧和最下侧各 10 个像素不会形成完整 patch。这是当前 586 输入设置本身的行为。

## 6. 投影与 SimFPN

### 6.1 图像通道投影

```text
[B,768,36,36]
→ 1×1 Conv(768→512)
→ [B,512,36,36]
```

### 6.2 文本通道投影

```text
[B,64,768]
→ Linear(768→256)
→ [B,64,256]
```

### 6.3 SimFPN

当前已经恢复到高性能备份实验的原始分支逻辑：SimFPN 完整输出 512 通道并执行跨 512 通道的 LN2d，随后检测分支取前 256 通道：

```text
x_c1: [B,512,144,144] → 前256通道用于检测
x_c2: [B,512,72,72]   → 前256通道用于检测
x_c3: [B,512,36,36]   → 前256通道用于检测
x_c4: [B,512,18,18]   → 前256通道用于检测
```

这些特征全部用于检测分支：

```text
[B,256,144,144]
[B,256,72,72]
[B,256,36,36]
[B,256,18,18]
```

多尺度总空间位置数：

```text
144² + 72² + 36² + 18² = 27,540
```

每个尺度还会生成：

```text
visual mask:       [B,Hl,Wl]
position embedding:[B,256,Hl,Wl]
```

visual mask 标记有效 resize 图像和 padding 区域。

AerialVG 没有 mask 标注，但当前完整保留并构建高性能备份版中的 SegHead、UNet decoder、MTD 和历史预测层。FPN 后 256 通道仍按原逻辑组成 `seg_feat`；因为 AerialVG 的 `target_mask=None` 且推理使用 `with_mask=False`，分割 decoder 不执行，所以 `loss_mask=0`。这样构建逻辑、参数布局、随机初始化顺序和数据流均与指定备份版本一致。

## 7. Deformable DETR 生成候选框

模型有 256 个可学习 query：

```text
query content embedding:  [256,256]
query position embedding: [256,256]
```

扩展到 batch：

```text
query_embed_input: [B,256,256]
query_pos_input:   [B,256,256]
```

Deformable DETR Decoder 有 3 层。每个 query 从四层 FPN 特征中进行多尺度可变形采样，输出：

```text
inter_states: [3,B,256,256]
```

含义：

```text
3   = decoder 层数
B   = 每卡 batch size
256 = query 数量
256 = query feature dimension
```

每层分别进行分类和框回归。

分类头：

```text
[B,256,256] → Linear → [B,256,2]
```

两个类别为 foreground 与 no-object/background。

框回归头：

```text
[B,256,256] → MLP → [B,256,4]
```

经过 sigmoid 后：

```text
outputs_class: [3,B,256,2]
outputs_coord: [3,B,256,4]
```

框使用相对有效图像归一化的：

```text
[cx,cy,width,height]
```

最终 decoder 层提供：

```text
候选特征：[B,256,256]
候选框：  [B,256,4]
类别：    [B,256,2]
```

## 8. 检测训练监督

pipeline 提供 resize 后的像素 `xyxy` 框：

```text
gt_bbox: List[[Ni,4]]
```

`prepare_targets()` 根据有效图像尺寸归一化：

```text
[x1/W,y1/H,x2/W,y2/H]
```

然后转为：

```text
[cx,cy,width,height]
```

得到：

```text
targets[i]["boxes"]:  [Ni,4]
targets[i]["labels"]: [Ni]
```

最终 decoder 层的 256 个预测与 `Ni` 个训练框进行 Hungarian matching。matching cost 为：

```text
classification cost × 1
L1 bbox cost        × 5
GIoU cost           × 2
```

例如一个样本有一个 refer GT 和四个 teacher 框，Hungarian matching 会从 256 个 query 中选出五个，分别匹配这五个目标。

其中 target index 0 始终是 refer GT。例如：

```text
query 17  → target 0（refer GT）
query 31  → target 1（teacher）
query 102 → target 2（teacher）
```

当前三个 decoder 层都计算：

```text
loss_class × 1
loss_bbox  × 5
loss_giou  × 2
```

随后：

```text
loss_det = 三层 DETR loss 之和 × 0.1
```

按照当前要求，三个 decoder 层继续共享最终层 Hungarian matching，这是保留的原始 PropVG 行为。

## 9. CRS 选择文本指向的候选框

CRS 输入：

```text
query features: [B,256,256]
text features:  [B,64,256]
text mask:      [B,64]
```

### 9.1 Query 对文本做 cross-attention

每个候选 query 对 64 个文本 token 做三层 cross-attention：

```text
[B,256,256] + [B,64,256]
→ [B,256,256]
```

### 9.2 句子级相关性

有效文本 token 做 max pooling：

```text
[B,64,256] → [B,256] → [B,1,256]
```

每个 query 与句子特征做 cosine similarity：

```text
sentence similarity: [B,256,1]
```

### 9.3 词级相关性

每个 query 与全部文本 token 做 cosine similarity：

```text
word similarity: [B,256,64]
```

取最相关的 token：

```text
word max score: [B,256,1]
```

### 9.4 融合得分

通过学习权重融合句子级与词级得分，再经过 sigmoid：

```text
refer_score: [B,256,1]
```

### 9.5 Refer loss

根据 Hungarian matching 找到与 target 0，也就是原始 refer GT 匹配的 query：

```text
target_gt_refer: [B,256]
```

每张图只有一个 referring 正样本：

```text
匹配原始 refer GT 的 query = 1
其他 query                 = 0
```

匹配 teacher 框的 query 在 refer loss 中仍是负样本，因为 teacher 框只提供 proposal 检测监督，不是文本需要定位的目标。

```text
loss_refer = BCE(refer_score,target_gt_refer) × 1.0
```

当前总训练 loss 为：

```text
loss = loss_det + loss_refer
```

其余项：

```text
loss_mask   = 0：AerialVG 不运行分割分支
loss_nt     = 0：当前 AerialVG forward 没有调用 MTD
loss_filter = 0：当前 enable_grab=False
loss_crv：当前 enable_crv=True，监督反事实支持度与真/假候选排序
```

## 10. 推理与最终目标框

推理阶段不会把 GT 框传给模型。模型最终得到：

```text
outputs_class[-1]: [B,256,2]
outputs_coord[-1]: [B,256,4]
refer_score:       [B,256,1]
```

归一化的 `cxcywh` 框先转为 `xyxy`，再乘有效 resize 图像宽高：

```text
x 坐标 × img_shape.width
y 坐标 × img_shape.height
```

输出：

```text
pred_boxes:  [B,256,4]
pred_scores: [B,256]
```

当前：

```python
with_nms = False
```

候选框最终只按照 CRS 的 `refer_score` 排序。检测分类概率会被保留为 `det_score`，但不参与 AerialVG 的最终候选排序。

```text
Top-1 框 = refer_score 最高的候选框
Top-5 框 = refer_score 最高的前五个候选框
```

指标计算：

```text
Top-1 Acc@0.5 = 最高分框与原始 GT 的 IoU >= 0.5
Top-5 Acc@0.5 = 前五框至少有一个与原始 GT 的 IoU >= 0.5
box mIoU      = 最高分框与原始 GT 的平均 IoU
```

验证和测试只使用原始 refer GT 计算指标，teacher 框不进入验证或测试。

## 11. Grab 开启时的可选流程

当配置改为：

```python
enable_grab = True
```

Deformable DETR 前面的过程不变，仍先产生：

```text
queries: [B,256,256]
boxes:   [B,256,4]
```

### 11.1 文本条件候选过滤

Grab 对 256 个候选产生：

```text
filter_logits: [B,256]
```

当前保留比例为 0.5：

```text
selected_indices: [B,128]
selected queries: [B,128,256]
selected boxes:   [B,128,4]
```

### 11.2 局部外观采样

每个保留候选框生成 48 个相对采样点：

```text
sampling_points: [B,128,48,2]
```

分别从四层 FPN 特征采样：

```text
每层 sampled feature: [B,128,48,256]
```

四层融合并通过 appearance attention 后：

```text
appearance feature: [B,128,256]
```

采样坐标会利用每层 visual mask 计算 valid ratio，将相对于有效图像的 box 坐标映射到含 padding 的特征图坐标。

### 11.3 候选拓扑图

128 个候选框两两构造七维几何关系：

```text
geometry:  [B,128,128,7]
adjacency: [B,128,128]
```

每个候选连接六个近邻，经过四层图消息传播：

```text
[B,128,256] → [B,128,256]
```

### 11.4 Grab 后的 CRS

增强后的 128 个 query 进入 CRS：

```text
[B,128,256] + [B,64,256]
→ refer_score [B,128,1]
```

最终输出 128 个候选框，而不是 256 个。

检测 loss 仍在 Grab 过滤前的全部 256 个 query 上计算。Grab 的 `loss_filter` 监督哪些候选应该被保留：所有 Hungarian 匹配到前景 GT 的 query 都是过滤正样本，未匹配 query 是负样本。

当前配置 `enable_grab=False`，所以上述 Grab 分支不会执行。CRV 直接从 PropVG/CRS 的 256 个候选中选择原始 Top-8 并反事实重排，最终只输出 8 个候选。CRV 细节见 `CRV_DESIGN.md`。

## 12. 当前实际主线总结

```text
图像 + 完整文本
    ↓
BEiT-3 联合视觉语言编码
    ↓
SimFPN 生成四尺度视觉特征
    ↓
Deformable DETR 生成 256 个候选框
    ↓
CRS 使用完整文本给 256 个候选框打 referring score
    ↓
CRV 选取原始 Top-8，用共享两层跨模态评分器分别计算原始场景与逐候选遮挡场景的支持度
    ↓
用“原始支持度减去遮挡后支持度”得到反事实重要性，乘法加权原始 CRS 分数并重排
    ↓
加权分数最高的候选框作为最终 visual grounding 结果
```

训练时，原始 refer GT 同时提供检测监督和 referring 监督；teacher 框只提供候选目标检测监督。验证和测试只使用原始 refer GT。
