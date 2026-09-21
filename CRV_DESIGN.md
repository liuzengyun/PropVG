# CRV：当前实现与数据流

## 1. 作用与位置

CRV（Counterfactual Referent Verifier）位于 CRS 初始评分之后、最终候选排序之前。它不替代 CRS，而是比较：

```text
原始场景对完整文本的支持度
        -
遮挡某一候选后的场景对同一完整文本的支持度
        =
该候选的 counterfactual drop
```

drop 越大，说明该候选消失后文本越难成立，因此它越可能是不可替代的 referring target。

当前配置入口是 `configs/aerialvg/PropVG-aerialvg.py`。CRV 和 Grab 是两个独立开关；当前配置同时开启二者，并从 CRS 原始分数中取 Top-5。关闭 Grab 时，CRV 仍可直接处理 PropVG 的 256 个候选。

## 2. 输入与尺寸

```text
最高分辨率 SimFPN 检测特征： [B,256,Hf,Wf]，586 输入时通常为 [B,256,144,144]
该层 padding mask：            [B,Hf,Wf]
待重排候选框：                 [B,N,4]，归一化 cxcywh
CRS referring score：          [B,N,1]
完整文本 token 特征：           [B,L,256]
文本 padding mask：             [B,L]
训练阶段 refer GT：             [B,4]，只用于 CRV loss，不参与验证或推理
```

`detach_inputs=True` 时，视觉、文本、候选框和 CRS 分数均在 CRV 入口截断梯度。CRV loss 只训练 CRV 自身。训练循环还会分别裁剪 PropVG 与 CRV 参数的梯度，避免 CRV 梯度通过全局梯度范数间接改变 PropVG 的更新幅度。

## 3. 候选选择

CRV 严格按原始 CRS 分数选择 Top-K：

```text
PropVG 无 Grab：DETR 256 → CRS 256 → CRV Top-K
启用 Grab：    DETR 256 → Grab 筛选/绑定 → CRS → CRV Top-K
```

推理只重排这 K 个候选，其余候选舍弃，不会出现原始第 K+1 名在重排后重新进入的问题。GT 从不被塞入推理候选集合。

## 4. 候选级视觉干预

候选框先根据有效图像宽高比例从有效图坐标映射到含 padding 的最高分辨率 FPN 坐标。每个通道在整张有效特征图上分别计算空间均值，遮挡区使用对应通道的均值填充，而不是置零，也不是对 256 个通道求共同均值。

每个候选独立生成一个矩形软 mask：候选框至 1.5 倍范围严格为 1，全部替换；1.5–2.0 倍范围用 smoothstep 从 1 平滑衰减至 0；2.0 倍范围外保持原特征。mask 不读取其他候选框，也不使用额外的“参考物体保护”规则。

这符合最初的直接反事实设计：测试“删除当前候选所在视觉区域”造成的整体语义变化。相应限制是，候选扩张区域如果覆盖了邻近参照物，参照物特征也可能被部分替换。当前实现接受这个限制，不额外引入依赖检测质量的保护逻辑。

padding 区先填成通道均值，且 verifier 使用 padding mask，所以 padding 特征不会成为虚假的语义证据。

## 5. 原始场景与 Mask 场景共同评分

每个高分辨率场景在完成干预后才池化到 24×24，再经 1×1 投影：

```text
[B,256,Hf,Wf]
→ 原始场景 + K 个候选遮挡场景
→ adaptive average pooling 24×24
→ visual_proj
→ [B,K+1,256,24,24]
```

训练时额外加入一个“精确 refer GT 遮挡场景”，只为在 Top-K 没有合格正候选时仍提供稳定的正监督：

```text
训练：[B,K+2,256,24,24] = 原始 + K候选 + 1个GT
推理：[B,K+1,256,24,24] = 原始 + K候选
```

所有场景和同一份完整文本进入共享的两层、八头 Transformer Decoder。视觉 token 为 24×24=576 个；查询是 1 个 support token 加 L 个完整文本 token。评分头输出未经 sigmoid 的 support logit：

```text
drop_i = original_support_logit - masked_support_logit_i
```

这里只学习成对差值，不要求绝对 support 数值具有概率意义。

## 6. CRV 训练监督

refer GT 始终是 AerialVG 原始定位目标，不是 teacher 框。候选按其与 refer GT 的 IoU 分组：

```text
IoU >= 0.5：正候选
IoU < 0.3 且候选2倍框覆盖refer GT不足10%：可靠 hard distractor
其余：忽略
```

当前相对损失包括：

```text
正候选：       relu(positive_margin - drop_positive)
精确GT遮挡：   relu(positive_margin - drop_gt)
可靠负候选：   smooth_l1(drop_negative, 0)
差值排序：      relu(margin - drop_positive + drop_negative)
GT差值排序：    relu(margin - drop_gt + drop_negative)
最终融合排序： 直接约束融合后的正确候选高于错误候选
最终平衡BCE：  直接监督推理时使用的融合logit
```

完全不重叠且扩张区域也不会覆盖 refer GT 的高 CRS 候选属于可靠负候选。随机低分背景不进入 CRV；负样本来自 CRS 已认为可能正确的 Top-K hard distractors。

内部总损失为：

```text
L_drop = L_candidate_positive + L_candidate_negative
         + gt_drop_loss_weight × L_exact_gt_positive

L_rank = L_candidate_rank + L_exact_gt_rank

L_crv_internal = L_drop
                 + ranking_loss_weight × L_rank
                 + fused_ranking_loss_weight × L_fused_rank
                 + final_bce_loss_weight × L_final_bce
```

`L_final_bce` 使用与评估一致的候选 IoU 标签：IoU≥0.5 为正，可靠的 IoU<0.3 候选为负，中间区域忽略。正、负 BCE 分别平均后再取均值，避免负候选更多而主导训练。如果一个样本的原始 CRS Top-K 中没有正候选，该样本跳过最终 BCE，因为压低所有候选不能找回不在集合中的目标。精确 GT mask 仍训练 drop，但没有对应的 CRS 候选分数，所以不参与最终 BCE。

当前 `REFHead` 再乘配置中的 `loss_weight["crv"]=0.5` 后加入总训练损失。绝对 `sigmoid(support_logit)` 不作为监督目标，也不参与融合；真正有意义的是同一评分网络对原始场景和 mask 场景产生的 logit 差值。训练日志额外显示 `crv_final_bce` 和 `crv_alpha`，二者只是观测指标，不会被重复加入总损失。

## 7. 最终融合

CRV 不会丢弃原 CRS 分数。它作为原始 CRS logit 上的有界残差：

```text
alpha = alpha_max × sigmoid(learnable_parameter)
0 < alpha < alpha_max

raw_logit_i = logit(clamp(raw_CRS_score_i, 1e-6, 1-1e-6))
residual_i = alpha × tanh(drop_i / temperature)
final_logit_i = raw_logit_i + residual_i
final_score_i = sigmoid(final_logit_i)
```

drop>0 会提升候选，drop<0 会抑制候选。`tanh` 将单个候选的修正限制在 `(-alpha,+alpha)`，而 `alpha_max=2` 又限制了全局融合强度，防止异常 drop 无限压过 CRS。`alpha` 初始为 1。support head 最后一层以全零初始化，因此训练开始时所有 drop 和 residual 精确等于 0，最终分数与原 CRS 完全一致，随机初始化不会先扰乱结果。

代码仍输出 `importance_scores=exp(residual)` 作为 odds multiplier 诊断量，但最终概率并不是简单的 `CRS×importance`，而是由上述 logit 残差公式得到。

完整推理顺序为：

```text
CRS 对 N 个候选正向评分
→ 按原始 CRS 分数保留 Top-K
→ 为原始场景和每个候选 mask 场景计算共享 support logit
→ 得到 K 个 drop 和 K 个有界 logit residual
→ 加到原始 CRS logits 上并执行 sigmoid
→ 下游按融合分数排序并输出目标框
```

当前 `K=5`，所以 Top-1 从这 5 个融合候选中取最高分，Top-5 检查这 5 个候选中是否至少一个框与 refer GT 的 IoU≥0.5，box mIoU 使用最终 Top-1 框计算。若将 `crv_topk` 设得小于 5，Top-5 指标就不再具备原定义下的五候选覆盖范围。

## 8. GT、teacher 与推理边界

- 原始 refer GT：用于 PropVG 原有监督、CRV 正负候选划分和训练期精确 GT 干预。
- teacher/辅助框：只属于训练阶段的 DETR proposal 检测监督，不会替换 refer GT。
- 验证和测试：CRV 不接收 GT，不构造精确 GT 场景，只用图像、完整文本和模型预测。

## 9. 当前配置参数

```python
enable_crv = True
crv_topk = 5

crv = dict(
    topk=crv_topk,
    verifier_dim=256,
    grid_size=24,
    num_heads=8,
    num_layers=2,
    verifier_dropout=0.0,
    core_expansion=1.5,
    feather_expansion=2.0,
    neutral_mode="mean",
    positive_drop_margin=0.5,
    margin=0.2,
    ranking_loss_weight=1.0,
    fused_ranking_loss_weight=1.0,
    final_bce_loss_weight=0.5,
    gt_drop_loss_weight=1.0,
    fusion_alpha_init=1.0,
    fusion_alpha_max=2.0,
    fusion_temperature=1.0,
    positive_iou_threshold=0.5,
    negative_iou_threshold=0.3,
    negative_max_gt_coverage=0.1,
    detach_inputs=True,
)
```

## 10. 代码位置

- CRV 核心：`propvg/models/heads/crv.py`
- 接入 CRS、训练 GT 与推理：`propvg/models/heads/REFHead.py`
- 独立梯度裁剪：`propvg/apis/train.py`
- AerialVG 配置：`configs/aerialvg/PropVG-aerialvg.py`
- 分数/遮挡可视化：`tools/visualize_crv_scores.py`

## 11. 已完成的验证与仍需实测的内容

已通过 Python 编译检查和 CPU 前向/反向单元测试，覆盖训练、无 GT 推理、输出维度、CRV 梯度存在性，以及零初始化时最终分数与原 CRS 分数完全一致。

当前 propvg 环境中的 PyTorch 仍报告 CUDA 不可用，因此本次修改后尚未进行真实 AerialVG GPU batch、显存、速度和精度回归测试。正式长训前应先恢复 PyTorch 的 CUDA 访问并跑一个 batch，再用小验证集确认 Top-1/Top-5 与 mask 可视化。
