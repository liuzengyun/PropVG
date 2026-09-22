# CRV：基于完整文本的反事实候选重排序

本文档对应当前 `propvg/models/heads/crv.py`、`propvg/models/heads/REFHead.py` 和 `configs/aerialvg/PropVG-aerialvg.py` 的实际实现。

## 1. 核心思路

CRS 回答正向问题：

```text
候选存在时，它与完整文本有多匹配？
```

CRV（Counterfactual Referent Verifier）位于 CRS 之后，回答互补问题：

```text
如果把候选对应的视觉证据从场景特征中抹去，完整文本的场景支持度下降多少？
```

对候选 `i`：

```text
drop_i = S(original_scene, full_text)
         - S(mask_candidate_i_scene, full_text)
```

`drop_i` 越大，表示候选被删除后文本越难由剩余场景支持。最终仍保留原始 CRS 分数，只把有界反事实差值作为 logit 残差：

```text
DETR proposals → CRS scores → CRS Top-K → CRV residual → fused scores
```

当前不使用候选 ROI 改写文本，不拆分目标词、属性词、关系词或参考物体词，也不读取 AerialVG 的辅助 region 作为 CRV 标签。

## 2. 当前输入与尺寸

当前 AerialVG 配置为 `img_size=586`、`num_queries=256`、`max_token=64`、`crv_topk=6`：

```text
最大 SimFPN 特征： [B,256,Hf,Wf]，通常约为 [B,256,144,144]
该层 padding mask：[B,Hf,Wf]
DETR 候选框：      [B,256,4]，归一化 cxcywh、相对有效 resize 图像
CRS 分数：          [B,256,1]
完整文本特征：      [B,64,256]
文本 padding mask： [B,64]
训练期 refer GT：  [B,4]，归一化 cxcywh
```

当前 `enable_grab=False`，因此 CRV 从 PropVG 的 256 个候选中直接选择 CRS Top-6。若以后打开 Grab，CRV 会在 Grab 输出的候选上执行相同流程。

## 3. 候选选择与训练期精确 GT 干预

CRS 原始分数只用于选择 Top-K：

```text
indices = topk(raw_crs_score.detach(), K=6)
```

选择索引不可导，CRV 不能通过 Top-K 操作改变哪些候选被选中。

训练时在 6 个预测候选之外额外加入一次精确 refer GT mask：

```text
训练 intervention：6 个预测候选 + 1 个精确 refer GT
推理 intervention：6 个预测候选
```

精确 GT 只为 `loss_counterfactual` 提供稳定正监督，不进入最终候选集合，不参与最终输出。验证和测试完全不把 GT 传给 CRV。

## 4. 坐标、padding 与 feature-level mask

预测框和 refer GT 都是相对有效 resize 图像归一化的 `cxcywh`。最大 SimFPN 特征对应的是 pad 后画布，因此先使用有效宽高比例映射：

```text
valid_ratio = [valid_width / padded_width,
               valid_height / padded_height]

center_on_canvas = center × valid_ratio
size_on_canvas   = size × valid_ratio
```

这保证长宽比缩放并 pad 到 `586×586` 后，mask 仍与目标在特征图上的位置一致。

每个视觉通道在有效图像区域内分别计算空间均值：

```text
neutral[b,c] = mean(visual[b,c,valid_pixels])
```

padding 区先用对应通道均值填充。候选区域也使用该通道均值替换，不是置零，也不是把 256 个通道混成一个均值。

软 mask 规则：

- 候选原框到 `1.5×` 框范围：mask=1，完全替换为通道均值；
- `1.5×` 到 `2.0×`：使用 smoothstep 羽化；
- `2.0×` 以外：mask=0，保留原特征。

干预先发生在最大空间特征上，然后池化：

```text
原特征/逐候选 mask 特征：[B,256,Hf,Wf]
adaptive average pooling
→ [B,256,24,24]
1×1 visual projection
→ [B,256,24,24]
```

训练时场景数为 `1+6+1=8`（原场景、6 个候选 mask、1 个 GT mask）；推理时为 `1+6=7`。

## 5. 完整文本与场景评分网络

同一份完整文本用于原场景和所有 mask 场景，不进行候选条件化：

```text
视觉 memory：24×24=576 tokens，每个 token 256 维
文本 query：  64 text tokens + 1 support token
```

视觉 token 加二维正弦位置编码。文本先经过线性投影，然后与一个可学习 support token 拼接。它们进入共享的两层、八头 Transformer Decoder：

```text
queries = [support_token; full_text_tokens]  # [65,256]
memory  = scene_visual_tokens               # [576,256]
decoded = SharedSceneVerifier(queries, memory)
support_logit = MLP(decoded_support_token)   # 标量
```

原场景只计算一次支持度，所有候选共享同一个 `S_orig`。verifier dropout 为 0，原场景和 mask 场景之间不会引入独立 dropout 噪声。

绝对的 `sigmoid(support_logit)` 不是概率监督目标；真正用于训练和融合的是同一评分器产生的 logit 差值 `drop`。

## 6. 训练标签

标签只由预测候选与原始 refer GT 的 box IoU 决定：

```text
IoU > 0.5：       正候选
IoU = 0：         完全不相交的负候选
0 < IoU <= 0.5：  忽略
```

代码使用严格的 `>`，因此恰好 `IoU=0.5` 的候选属于忽略区。评价指标仍按视觉定位标准使用 `IoU>=0.5`。

精确 refer GT intervention 始终作为 `loss_counterfactual` 的正样本。Teacher 框只参与 DETR proposal 检测监督，不替代 refer GT，也不直接进入 CRV 标签。

## 7. `loss_counterfactual`

定义：

```text
P = Top-K 中所有 IoU>0.5 的候选 + 精确 refer GT intervention
N = Top-K 中所有 IoU=0 的候选
d = support_drop_logit
m_pos  = 0.5
m_rank = 0.2
```

### 7.1 正样本差值约束

```text
L_pos = mean_{p∈P} ReLU(m_pos - d_p)
```

要求删除真实目标后支持度至少下降 `0.5` logit。

### 7.2 负样本不变约束

```text
L_neg = mean_{n∈N} SmoothL1(d_n, 0)
```

要求删除完全不相交的错误候选时，完整文本支持度尽量不变。

### 7.3 正负 pairwise 排序

```text
L_cf_rank = mean_{p∈P,n∈N} ReLU(m_rank - d_p + d_n)
```

每个正候选分别与每个负候选比较。多个正确 proposal 之间不使用 softmax，不会互相竞争。

```text
loss_counterfactual = L_pos + L_neg + L_cf_rank
```

即使 CRS Top-K 中暂时没有正确 proposal，精确 GT intervention 仍可通过 `L_pos` 和 `L_cf_rank` 训练 CRV。若某项没有有效样本，其值为保持计算图的 0。

该损失不依赖 CRS 分值，只更新 CRV 参数。

## 8. 融合与 `loss_final_rank`

CRV 不使用可学习 alpha，也不在验证集搜索融合系数：

```text
raw_logit_i = logit(clamp(raw_crs_score_i, 1e-6, 1-1e-6))
residual_i  = tanh(drop_i)                 # (-1,1)
final_logit_i = raw_logit_i + residual_i
final_score_i = sigmoid(final_logit_i)
```

`tanh` 把 CRV 对单个候选的修正限制在 1 logit 内。support head 最后一层全零初始化，因此初始化时：

```text
drop = 0
residual = 0
final_score = raw_crs_score
```

最终排序损失直接监督推理使用的 `final_logit`：

```text
L_final = mean_{p∈P_pred,n∈N} ReLU(m_rank - final_logit_p + final_logit_n)
```

其中 `P_pred` 只包含 Top-K 中的 `IoU>0.5` 预测候选，不包含额外 GT intervention。没有预测正候选或没有完全不相交负候选时，该样本无法组成 pair，`L_final` 跳过。

## 9. 梯度边界

当前目标是：

```text
loss_counterfactual → 只更新 CRV
loss_final_rank     → 只更新 CRV
```

进入 CRV 前，以下输入全部截断梯度：

```text
visual       = stopgrad(max_fpn_feature)
boxes        = stopgrad(predicted_boxes)
text         = stopgrad(full_text_feature)
refer_scores = stopgrad(raw_crs_scores)
```

因此：

- CRV 的两个损失都不能更新 CRS；
- CRV 也不能更新 `refer_adapter`、Grab、DETR、SimFPN、BEiT-3 或文本特征；
- CRS 只由原始 PropVG 的 `loss_refer` 训练；
- 训练和推理都只执行一次 CRS，使用同一套 CRS Top-K 和融合基线；
- `loss_final_rank` 仍可通过 residual 路径完整训练 CRV 自身。

训练循环把 CRV 参数和其余参数分组裁剪。CRV 自身的大梯度不会通过共同全局范数间接缩小 CRS 或其余 PropVG 参数的梯度。

## 10. 总损失与实际权重

```text
loss_crv_internal = loss_counterfactual + loss_final_rank
loss_crv_total    = 0.5 × loss_crv_internal

loss_total = loss_det
           + loss_refer
           + loss_filter
           + loss_mask
           + loss_crv_total
```

当前 AerialVG 配置中 `enable_grab=False` 且没有 mask GT，所以 `loss_filter=0`、`loss_mask=0`。

外部 `loss_weight["crv"]=0.5` 同时作用于两个 CRV 损失，但这些梯度只进入 CRV。原始 `loss_refer` 对 CRS 和前面网络的正常训练路径不变。

日志含义：

```text
loss_crv       = 0.5 × (crv_cf + crv_final_rank)
crv_cf         = 未乘 0.5 的 loss_counterfactual
crv_final_rank = 未乘 0.5 的 loss_final_rank
```

`crv_final_rank=0` 可能表示当前 batch 没有有效正负 pair，也可能表示所有有效 pair 已满足 0.2 margin，不等同于模块没有运行。

## 11. 训练与推理差异

| 项目 | 训练 | 验证/测试 |
|---|---|---|
| CRS 前向 | 原始前向一次 | 原始前向一次 |
| 精确 refer GT mask | 使用，仅训练 CRV | 不使用 |
| 候选 mask | CRS Top-6 | CRS Top-6 |
| Teacher/辅助框进入 CRV | 否 | 否 |
| 输出候选数 | 6（训练预测仅供内部使用） | 6 |
| 最终排序 | CRS logit + CRV residual | CRS logit + CRV residual |

验证和测试只用 GT 在模型外计算 Top-1 Acc@0.5、Top-5 Acc@0.5 和 top-1 box mIoU，不发生数据泄露。

## 12. 当前配置

```python
enable_grab = False
enable_crv = True
crv_topk = 6

crv = dict(
    topk=6,
    verifier_dim=256,
    grid_size=24,
    num_heads=8,
    num_layers=2,
    verifier_dropout=0.0,
    core_expansion=1.5,
    feather_expansion=2.0,
    neutral_mode="mean",
    positive_drop_margin=0.5,
    rank_margin=0.2,
    positive_iou_threshold=0.5,
    detach_inputs=True,
)
```

当前快速实验还使用：

```text
train = 官方 val：vg_val_odvg.jsonl + teacher_valset.json
val   = 官方 test：vg_test_odvg.jsonl
test  = 官方 test：vg_test_odvg.jsonl
```

这只是加速调试设置，`val` 与 `test` 完全相同，不是规范的最终论文划分；该设置会让基于 val 的模型选择等价于使用 test 选择模型。正式结果应恢复官方 train/val/test 独立划分。

## 13. 当前保留的限制

1. 描述中的参考物体如果进入 Top-K 且与 refer GT 完全不重叠，会被当成负候选；当前按要求不做参考物体保护。
2. 负候选框自身即使 `IoU=0`，其 `2×` 羽化 mask 仍可能覆盖邻近 refer GT；当前按要求不再使用扩张 mask 覆盖率过滤。
3. CRV 只能重排 CRS Top-6，正确目标不在 Top-6 时无法找回；当前不做候选去重或扩大 K。
4. 只使用最大 SimFPN 特征，不融合其他尺度。
5. support 绝对值未校准为概率，只有 `drop`、residual 和最终融合分数具有训练定义。
6. Top-5 是融合后 6 个候选中的前 5 个，因此 CRV 可能把原始 CRS 第 5 名与第 6 名互换，Top-5 既可能升也可能降。
7. 当前各损失对 batch 内所有有效候选或正负 pair 直接求全局平均，因此有效 pair 更多的样本权重也更大，而不是严格的逐图等权。

## 14. 代码位置与验证

- CRV 核心：`propvg/models/heads/crv.py`
- CRS/CRV 接入与梯度截断：`propvg/models/heads/REFHead.py`
- 损失汇总和独立梯度裁剪：`propvg/apis/train.py`
- 当前配置：`configs/aerialvg/PropVG-aerialvg.py`
- 三种分数评估：`tools/eval_crv_only.py`
- 原场景/GT mask/候选 mask 可视化：`tools/visualize_crv_scores.py`

已完成的代码级验证：

- Python 编译和 `git diff --check`；
- 合成 batch 训练、反向和无 GT 推理；
- `IoU>0.5`、`IoU=0` 和忽略区标签检查；
- 多个正候选均获得独立 pairwise 梯度；
- `loss_counterfactual` 不产生 CRS 分数梯度；
- `loss_final_rank` 不产生 CRS 梯度；
- CRV 不向视觉特征、候选框或文本输入回传梯度；
- 初始融合分数严格等于 CRS 分数；
- 原场景支持度在全部候选之间严格共享。

旧版候选条件文本 CRV 的 checkpoint 与当前结构、监督目标均不同，不能作为当前设计的公平结果；正式比较应重新训练。
