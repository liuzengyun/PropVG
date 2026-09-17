# CRV：基于反事实推理的候选优化

## 1. 目的

CRV（Counterfactual Referent Verifier）用于解决航拍视觉定位中的同质目标歧义。基础 CRS 回答：

```text
候选存在时，它与文本有多匹配？
```

CRV 补充回答：

```text
如果删除这个候选的视觉证据，剩余场景还能否支持完整文本？
```

删除正确目标后，剩余场景对文本的支持应下降；删除错误干扰候选后，真正目标仍然存在，文本支持应保持较高。

## 2. 开关与当前配置

配置文件：`configs/aerialvg/PropVG-aerialvg.py`

```python
enable_crv = True
crv_topk = 8

crv = dict(
    topk=8,
    verifier_dim=256,
    grid_size=24,
    num_heads=8,
    num_layers=2,
    verifier_dropout=0.0,
    core_expansion=1.5,
    feather_expansion=2.0,
    neutral_mode="mean",
    margin=0.2,
    ranking_loss_weight=1.0,
    positive_iou_threshold=0.5,
    negative_iou_threshold=0.3,
    negative_max_gt_coverage=0.1,
    detach_inputs=True,
)
```

当前设置 `enable_grab=False`，所以完整候选流为：

```text
Deformable DETR：256 candidates
        ↓
CRS：得到 256 个原始 referring scores
        ↓
CRV：选择原始 CRS Top-8
        ↓
反事实支持评分与融合重排
        ↓
最终只输出 8 candidates
```

CRV 不依赖 Grab。若 `enable_grab=False`：

```text
Deformable DETR 256 → CRS 256 → CRV Top-8 → 输出 8
```

## 3. 输入与维度

CRV 位于 CRS 之后，输入为：

```text
最高分辨率 FPN 特征： [B,256,144,144]
该层 padding mask：   [B,144,144]
候选框：               [B,N,4]，归一化 cxcywh
CRS referring score： [B,N,1]
完整文本特征：         [B,64,256]
文本 padding mask：    [B,64]
```

其中：

```text
N=128：Grab 开启时
N=256：当前 Grab 关闭
```

当前 `detach_inputs=True`，以上视觉特征、文本特征、候选框和原始 CRS 分数进入 CRV 时全部截断梯度。该操作不改变任何前向数值，只限制 `loss_crv` 的反向传播范围。

## 4. Ambiguous Candidate Set

根据原始 CRS referring score 选择 Top-K：

```text
indices = topk(refer_scores, K=8)
selected_boxes:  [B,8,4]
selected_scores: [B,8]
```

推理阶段严格只使用原始 Top-8，其余候选直接舍弃。CRV 重排后即使某个候选分数下降，也不会让原始第 9 名重新进入。

训练和推理均严格采用原始 CRS Top-8，不使用 GT 强行注入候选，避免训练/推理候选分布不一致。如果 Top-8 中没有同时出现 IoU 达标的目标候选和可靠干扰候选，该样本不计算 CRV loss；CRV 只学习有效的候选间反事实差异。

## 5. 空间特征与 padding

模块使用最高分辨率的 `[B,256,144,144]` FPN 特征。反事实遮挡直接发生在该分辨率上，避免航拍小目标在池化前已经与周边特征混合。每个候选依次在 144×144 上完成替换，然后才池化：

```text
[B,256,144,144]
→ 候选框平滑 mask 与均值特征替换
→ adaptive average pooling
→ [B,256,24,24]
→ 1×1 projection
→ [B,256,24,24]
```

该特征仍然来源于最高分辨率 FPN 层，而不是使用低分辨率 FPN 层。

根据 `[B,144,144]` visual mask 计算有效宽高比例：

```text
valid_ratio: [B,2] = [valid_width_ratio, valid_height_ratio]
```

DETR 框相对于有效图像归一化，而空间特征包含 padding，因此在生成反事实 mask 前，将候选框映射到 padded feature coordinates。该处理与 Grab 的 padding 修正一致。

## 6. 平滑扩张反事实 Mask

每个候选框使用两级扩张边界：

```text
mean_fill_core = box_size × 1.5
feather_outer  = box_size × 2.0
```

在原始 144×144 FPN 网格上生成“硬核心、软外沿”矩形 mask：

```text
soft_masks: [B,8,144,144]
```

原始候选框到 1.5 倍框以内的 mask 严格等于 1，全部替换为有效区域均值；1.5 倍到 2.0 倍框之间使用 smoothstep 平滑衰减到 0；2.0 倍框之外保持原特征。这样目标及紧邻边缘被完整抑制，过渡只发生在更外层区域。

预测框仅用于确定干预区域，进入 CRV 时会 `detach`，所以 CRV loss 不会通过 mask 坐标反向干扰 DETR 框回归。

## 7. 中性特征替换

当前使用：

```python
neutral_mode = "mean"
```

对每张图的有效视觉区域计算通道均值：

```text
neutral_feature: [B,256,1,1]
```

构造反事实特征：

```text
counterfactual = visual × (1-mask) + neutral × mask
```

输出：

```text
original_feature:       [B,1,256,24,24]
counterfactual_features:[B,8,256,24,24]
```

因此候选区域不是直接置零，而是替换为该图有效区域的平均语义特征。

也支持：

```python
neutral_mode = "learnable"
```

此时使用一个可学习的中性特征向量。

## 8. Counterfactual Referent Verifier

原始场景与反事实场景使用同一个评分器。视觉场景组成：

```text
[B,1,256,24,24] 原始场景
[B,8,256,24,24] mask场景
→ 合并为 [B,9,256,24,24]
→ 加二维正弦位置编码
→ [B×9,576,256]
```

完整文本投影为：

```text
[B,64,256]
→ Linear(256→256)
→ 加入一个可学习 support token
→ 针对原始场景和8个反事实场景复制
→ [B×9,65,256]
```

使用两层、八头 Transformer Decoder。每层先在 support token 与完整文本 token 之间做 self-attention，再对带二维位置的视觉 token 做 cross-attention：

```text
Q = support token + full text: [B×9,65,256]
K,V = visual scene tokens:     [B×9,576,256]
output:                        [B×9,65,256]
```

cross-attention 使用视觉与文本 padding mask，因此右侧、下侧图像 padding 和文本 padding 都不会参与支持度判断。评分器 dropout 为 0，避免原始/反事实配对因不同随机 dropout 产生伪差值。

取 decoder 输出的 support token，通过共享 MLP 分别得到原始场景和 mask 场景的支持度 logit：

```text
original_support_logit: [B,1]
masked_support_logits:  [B,8]
support_drop_logits = original_support_logit - masked_support_logits
```

解释：

```text
drop 高：删除候选导致支持度明显下降，候选更可能是真实目标
drop 低或为负：删除候选没有破坏文本支持，候选更可能是干扰项
```

## 9. 训练监督

原始 PropVG 检测和 CRS loss 保持不变：

```text
loss_det：对完整 256 个 DETR query 训练
loss_refer：对 Grab 后的候选（或无 Grab 时的 256 候选）训练原始 CRS
```

CRV 使用原始 refer GT 与 Top-8 预测框的 IoU 产生监督：

```text
IoU >= 0.5：覆盖真实目标，drop target = 1
IoU < 0.3 且2倍外框覆盖GT比例 < 0.1：可靠 hard distractor，drop target = 0
0.3 <= IoU < 0.5：忽略，不计算 CRV loss
```

因此，同一个真实目标上的重复 proposal 不会被错误地监督为 distractor。随机低分背景不会作为 CRV 的主要负样本，干扰项来自原始 CRS 的高分候选。如果训练早期没有任何 Top-8 框达到 0.5 IoU，该样本不产生排序监督。

### 9.1 支持度差值分类损失

对真实 refer 与 hard distractor 的支持度差值分别求 BCE 后等权平均：

```text
L_drop = balanced BCE(support_drop_logits, drop_target)
```

### 9.2 排序损失

要求删除真实目标造成的支持度下降，比删除干扰候选造成的下降至少高 margin：

```text
L_rank = max(0, margin - drop_true + drop_distractor)
margin = 0.2
```

模块内部：

```text
L_CRV_internal = L_drop + L_rank
```

在总模型中使用：

```text
loss_crv = 0.5 × L_CRV_internal
```

训练总损失：

```text
loss = loss_det + loss_refer + loss_filter + loss_crv
```

当前 Grab 关闭，所以 `loss_filter=0`。AerialVG 没有 mask，分割 decoder 和 MTD 不进入当前 forward，因此 `loss_mask=0`、`loss_nt=0`。

### 9.4 梯度边界

当前采用严格旁路训练：

```text
loss_det    → DETR、SimFPN、BEiT-3
loss_refer  → CRS、Grab、DETR、SimFPN、BEiT-3
loss_filter → Grab
loss_crv    → 仅 CRV 参数
```

`loss_crv` 不会修改基础候选框、基础视觉语言特征或原始 CRS 排序。这样 CRV 可以改善最终重排，但不会通过反向传播破坏基础 PropVG/Grab 模型。

## 10. 分数融合与最终输出

CRV 将原始场景与 mask 场景的支持度差直接转换为候选重要性：

```text
support_drop = support_original - support_masked
crv_importance = sigmoid(support_drop)
```

最终分数保留原始 CRS 正向匹配信息，并由 CRV 重要性进行乘法加权：

```text
final_score = original_CRS_score × crv_importance
```

效果：

```text
原始 CRS 高 + 删除后支持明显下降 → 保留较高分，更像真实目标
原始 CRS 高 + 删除后支持基本不变 → 被 CRV 权重抑制，更像同质干扰项
```

模型最终只输出：

```text
boxes:  [B,8,4]
scores: [B,8]
```

Top-1、Top-5 和 mIoU 评估代码不需要修改，会直接对融合重排后的 8 个候选计算指标。

## 11. 关键代码

```text
propvg/models/heads/crv.py
    CRV 模块、Top-K、视觉干预、跨模态评分、损失与融合

propvg/models/heads/REFHead.py
    在 CRS 后接入 CRV；同时兼容 PropVG 和 Grab

propvg/apis/train.py
    将 loss_crv 加入反向传播、日志和 W&B

configs/aerialvg/PropVG-aerialvg.py
    CRV 开关与超参数
```

## 12. 消融开关

纯 PropVG：

```python
enable_grab = False
enable_crv = False
```

PropVG + CRV：

```python
enable_grab = False
enable_crv = True
```

Grab：

```python
enable_grab = True
enable_crv = False
```

Grab + CRV：

```python
enable_grab = True
enable_crv = True
```
