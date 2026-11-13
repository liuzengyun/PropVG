# 航拍视觉指代的文本条件竞争集合与关系绑定细化（BIRS）

## 1. 当前方案与动机

当前新实验使用 **Top-N 关系细化 + 全候选融合 BCE**。只为 CRS Top-N 计算绑定残差，但对全部有效候选计算融合定位 BCE；不再维护此前的聚类/Top-N-only-BCE BIRS 兼容路径。BIRS 仍为 Binding Interaction Residual Scoring（关系绑定交互残差评分），放在 CRS 之后，不替换 CRS，不修改预测框，不重跑图像编码器或检测解码器。

航拍小目标的细粒度外观信息有限，密集场景中经常存在颜色、类别、尺度相似的实例。第一阶段匹配可以找到“像描述的一组目标”，但不一定确定具体实例。广域俯视/斜俯视图像又允许观察较多物体及其配置，因此参考物的语义与相对位置可能提供补充证据。

核心问题不是分别判断“有黑车”和“左侧有物体”，而是判断“黑车是否绑定在左侧”。实际绑定相对于重绑定对照的增益，用于补充原图文匹配。

**为何不用外观聚类？** 外观特征相似不等于竞争当前描述。上一轮 epoch34 诊断中，test 有832条“CRS错误但Top-5含正确候选”的样本，正确候选与错误Top-1同簇仅271条。新方案直接用 CRS Top-N 定义文本条件下的竞争集合，避免聚类边界限制绑定排序监督。

Top-N 筛选和中心化本身不是创新；关系绑定与条件竞争集合的组合需要消融和查新验证，不能保证原创或性能提升。这是针对航拍困难的设计，不是只适用于无人机的数学机制；重绑定只是统计关联对照，不是已识别的真实因果效应。

## 2. 与现有模块的关系

- 原 CRS 始终保留，原定位损失照常监督。
- 可选 Grab / BCEM 的原流程保持；BCEM 在 CRS 前增强表示，BIRS 不读取其空间路径或图像特征图。
- Grab 若筛掉部分目标候选，参考支路仍读原检测池的 refer_adapter 特征，不从 Grab 筛选结果裁剪参考。
- 不需要 teacher、关系标签、额外参考物 GT、文本人工拆解或外部模型。
- BIRS 与旧 CRV / two-stage relation refinement 互斥，避免多个重排器叠加。
- 关闭 enable_birs 时不构建新模块，不增加损失、参数或collective。

## 3. 完整数据流与维度

B为批大小，M为进入CRS的候选数，P为原检测候选数，L为文本长度；常用M=P=256、L=64、C=256。K=min(target_topn,M)，默认K=5；参考上限R=64，内部D=128。不足K个有效框时只使用实际有效候选，不复制或注入GT。

```text
BEiT-3 → SimFPN → Deformable DETR
                       ↓
              queries [B,P,256] + 框 + 检测分类
                       ↓
              可选Grab → refer_adapter → 可选BCEM
                       ↓
              原CRS(Q [B,M,256], T [B,L,256])
                       ↓
               原logits / 概率 [B,M,1]
                       ↓
             按原CRS概率取Top-N [B,K]
                       ↓
         目标特征 [B,K,256]、框 [B,K,4]
                       ↓
             平均原型 [B,1,256] → [B,1,128]

原检测池 → 检测置信度Top-64 → 去重 → R个槽位/有效掩码
                       ↓
        参考特征投影 → 读取完整文本 → [B,R,128]
                       ↓
       原型条件的参考语义 H [B,1,R,128]

Top-N目标框 × 参考框 → 相对几何 [B,K,R,9]
                → 文本/原型条件非线性编码 Ψ [B,K,R,128]
                       ↓
             实际绑定 − 重绑定期望 [B,K]
                       ↓
         有效集合内中心化 × 可学习β → 残差 [B,K]
                       ↓
            修正Top-N内部logit并排序
                       ↓
       重排后的Top-N + 原CRS其余候选的原顺序
```

实现只为K个目标计算绑定，不为全部M个目标计算；全部M个输出槽位仍保留用于旧后处理。raw_binding、delta等诊断量回填到原query索引，集合外为零，candidate_mask标明计算范围。参考池始终独立，不随K缩小。目标侧关系计算量约O(BKRD)，参考文本读取约O(BRLD)；没有R×R显式置换矩阵。分块只控制单次临时工作集，训练反向仍保留各块所需激活。

## 4. 每一步的目的与精确规则

### 4.1 文本条件竞争集合与共享原型

按原评估的有效框过滤、CPU概率降序排序规则取Top-N。选择索引停止梯度；选中特征的梯度按detach_inputs开关决定。没有重新训练一个筛选器，不在训练时加入GT候选。

Top-N特征平均得到一个共享原型z，代替外观聚类原型。目的：让同一条描述的实际竞争者使用一致的参考语义条件。默认不再执行K-means，也不依赖num_groups或kmeans_iters。

### 4.2 独立参考池

检测前景置信度为softmax(class_logits)[...,0]。有效检测框先取Top-64，再按置信度贪心抑制IoU≥0.8的近重复框，不从Top-64外补齐。

对每个目标排除同源query及IoU≥0.8的自身近重复；真实部分重叠允许。框从相对有效图像的cxcywh转换为xyxy并裁剪，几何不用pad_shape。选择、框坐标和检测分数停止梯度。

目的：参考物自身不必像目标，不能也用CRS Top-N作为参考池。

### 4.3 完整文本与语义—几何编码

参考特征与完整文本投影到128维，参考token读取有效文本token，得到v_j。padding不参与；全padding样本不产生有效残差。

参考语义：
$$H_j=\tanh\{\operatorname{LN}[\operatorname{GELU}(W_h[v_j;z;v_j\odot z])]\}.$$

对目标i和参考j，构造9维几何：
$$r_{ij}=[\Delta_x,\Delta_y,\Delta_x/d,\Delta_y/d,\log(1+d),
\log(w_j/w_i),\log(h_j/h_i),\tanh((x_j-x_i)/w_i),\tanh((y_j-y_i)/h_i)].$$

前两维为像素中心差除以有效图像短边，d为该归一化距离；方向坐标向右、向下为正。宽高比对数截断[-8,8]，分母有数值下限。非线性MLP编码后，与“原型+有效文本投影平均值”交互，输出Ψ_ij。

目的：保留“什么物体在什么相对位置”的绑定。不能只线性编码dx、dy：在参考维度中心化时目标中心会被消掉，导致不同目标难以区分。

### 4.4 实际配对减重绑定期望

对目标i的同一个合法参考集合J_i，m_i=|J_i|：
$$p_i=\frac{1}{m_i\sqrt D}\sum_{j\in J_i}H_j^\top\Psi_{ij},\qquad
n_i=\frac{\bar H_i^\top\bar\Psi_i}{\sqrt D},\qquad d_i=p_i-n_i.$$

等价计算：
$$d_i=\frac{1}{m_i\sqrt D}\sum_{j\in J_i}(H_j-\bar H_i)^\top(\Psi_{ij}-\bar\Psi_i).$$

n_i是均匀随机重绑定的精确期望，无需抽样。两侧均值使用相同合法参考集合。若分数只有独立的语义项+位置项，则残差为零；参考语义完全相同时也应为零。少于两个合法参考时不产生绑定。没有把所有置换标成负样本，也没有删除真实物体。

### 4.5 残差与硬边界排序

至少两个目标候选能产生合法绑定时，在这些有效候选内中心化：
$$\delta_i=\beta(d_i-\bar d),\quad\beta=\beta_{max}\sigma(a),\quad
\ell_i^{final}=\ell_i^{CRS}+\delta_i.$$

β初始0.1，上界2，仍通过损失学习；没有新增门控。无有效绑定的候选δ=0。中心化减去共同常数时不改变这些候选内部的排序，主要限制平均logit漂移、影响BCE校准，不应把它当作额外消歧能力。

最终用：
**Top-N内按修正logit重排 + 集合外按原CRS顺序接在后面。**

通过现有ranking_priority接口编码完整顺序。不能简单对所有修正后的概率再次排序，否则集合外候选仍可能插进来。概率仍是sigmoid(logit)，ranking_priority是序数，不是置信度；评估/可视化必须优先使用ranking_logits/ranking_priority。

N=5保证同一checkpoint重排前后Top-5成员相同，因而其Top-5 Acc相同；N>5允许改变Top-5，但Top-N成员不变。联合训练仍可能改变CRS，不能保证相对于独立BASE的Top-5不变。

## 5. 损失、监督与梯度

原loss_det、loss_refer、可选Grab/filter/mask/correctness保持原算法与权重，原CRS损失仍对全部原候选计算。

| 新增损失 | 默认权重 | 范围与目的 |
|---|---:|---|
| loss_birs_final | 1.0 | 全部有效候选上的融合BCE，沿用原Hungarian refer标签，直接监督最终定位 |
| loss_birs_binding | 0.1 | Top-N内合法正负对，softplus(d_neg-d_pos)，独立训练绑定分支 |

日志值已乘权重。设全部候选有效掩码为v_i，原Hungarian refer标签为y_i，融合BCE为：

$$\mathcal L_{final}^{(b)}=\frac{\sum_{i=1}^{M}v_i\operatorname{BCEWithLogits}(\ell_i^{final},y_i)}{\max(1,\sum_{i=1}^{M}v_i)}.$$

再按全局有效样本数平均。**分母是全部有效候选数（通常256），不是Top-N数；有效样本也不要求正候选在Top-N。**只要求存在原refer GT且其原匹配正候选在全部有效框中。不因没有绑定参考而跳过融合BCE。

集合外候选的最终logit就是原CRS logit，因此GT匹配正候选在前五之外时仍得到直接定位梯度，前五中的错误候选也仍受负标签监督。此时绑定分支没有集合外正候选的残差计算，不能把“全候选BCE”解释为“全候选绑定正例监督”。无合法绑定正负对时仅跳过绑定排序损失，不跳过融合BCE。

绑定正候选仍须原Hungarian正标签、IoU≥0.5且可产生合法绑定；负候选须与原refer GT的IoU恰为0且有效，按原CRS得分最多取negative_topk个。其他候选不进入此排序项；没有合法对则跳过。共享集合替代同簇限制，其他正负规则不变。

仅loss()读取原refer GT；teacher或辅助GT不作为BIRS关系标签。DDP按全局有效样本数归一化，缺监督rank也保留新参数的连接零梯度图。

- detach_inputs=False：联合训练；新增损失可回传原CRS及候选/文本特征路径，框和离散选择无直接梯度。
- detach_inputs=True：候选、参考、文本及原CRS概率/logit截断，仅训练BIRS；梯度裁剪独立，原损失继续训练原模型。

本次修正撤销了此前“融合BCE只在Top-5内平均”的变化，恢复全候选监督范围、候选维度分母和全局有效样本覆盖，不额外调损失权重。残差很小时仍相当于额外增加一次定位监督；相对于不含BIRS的BASE，训练目标并不相同，收益不能直接归因为绑定机制。

## 6. 配置、训练与旧模型兼容

当前配置：

```python
enable_birs = True          # False完全沿用原模型
birs_target_topn = 5
birs = dict(
    target_topn=birs_target_topn,
    reference_topk=64, duplicate_iou=0.8,
    model_dim=128, num_heads=4, chunk_size=32,
    initial_strength=0.1, max_strength=2.0,
    final_loss_weight=1.0, binding_loss_weight=0.1,
    negative_topk=8, detach_inputs=False,
)
```

原CRV和relation_refinement保持关闭。启动方式不变，建议设置独立work_dir：

```bash
conda activate propvg
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/aerialvg/PropVG-aerialvg.py
```

兼容范围：

- enable_birs=False（旧配置没有此参数时也默认False）：仍沿用原PropVG/Grab/BCEM/correctness等已有路径，既有非BIRS模型可以继续训练/测试，不要求新增BIRS参数。
- 当前BIRS只保留Top-N计算路径，移除candidate_mode、num_groups、kmeans_iters以及聚类实现。旧BIRS配置可能直接报未知参数；不维护修改前版本和1113v1的生产代码兼容。
- 当前schema=3，策略摘要包含全候选损失口径和target_topn。旧BIRS权重不能直接用当前配置--load-from/--resume-from，也不应以恢复训练代替新对照。
- 归档实验源码、归档配置和权重没有删除；如需要复查旧实验，可在对应归档环境运行，而非要求当前代码复现旧BIRS。
- 从相同BEiT-3初始化重新训练本组对照。需要基线权重初始化时，可使用不含BIRS的baseline配合--finetune-from；完整新BIRS训练后仍保存原模型+BIRS的整体参数。
- 未完成有效监督优化的BIRS在eval时回退原CRS。这里trained_steps记录有新增监督的优化步，不保证每一步都有绑定正负对。

## 7. 诊断接口、日志与验证

核心文件：propvg/models/heads/binding_residual.py；接入点仍是REFHead._binding_residual()。保持原诊断接口：

```python
with model.head.birs.diagnostics():
    # 正常前向后查看extra_dict["birs_samples"]
    ...
```

新增candidate_mask、target_indices、target_valid，明确哪些原query参与Top-N。其他字段保留全候选原索引，包括原分数、最终logit、raw_binding、paired/null、delta、组标记（选中为0，未选中为-1）、参考池与合法参考掩码。

日志含义：birs_supervised_samples现在是全局每batch可计算全候选融合BCE的样本数，正常32条有refer GT且匹配正框有效时应接近32，不再等于32×refer_in_topn。birs_binding_samples仍统计有合法绑定对的样本数。birs_target_count与birs_refer_in_topn分别表示计算绑定的目标数、原匹配正候选进集合的比例（不是IoU Top-N召回率）。原birs_eligible与delta_abs仍按全部候选槽位平均，Top-N版数值自然更小，不应误判为分支关闭。

验证重点：

1. N=5时每图前后Top-5集合相同、集合外顺序不变，极端残差也不能让外部候选插入。
2. 只对Top-N执行几何/绑定，不运行聚类；参考物仍能来自Top-N以外。
3. 匹配正候选在Top-N之外时，全候选融合BCE仍非零且对集合外正logit有梯度，只有绑定排序损失跳过；分母仍为全部有效候选。
4. 不维护旧BIRS兼容；验证False路径的旧参数/RNG/前向/梯度不受影响，既有非BIRS权重可严格加载。
5. 统计与独立BASE的整体性能，以及同checkpoint前后改对/改错；联合训练时二者含义不同。
6. 记录全候选融合监督样本数、匹配正候选Top-N覆盖率、合法绑定对比例、β、残差和CRS分差，不能只看loss。\n7. 无可用参考物或全padding文本时残差为零，但只要存在有效refer正框，融合BCE仍计算；GT不存在或无有效匹配正框时跳过新增监督。
8. 保留paired与crs_control对照；crs_control残差/绑定loss为零，只保留新增全候选BCE，可控制额外定位监督的影响。BASE、本控制和完整模块需同配置独立训练，不能只靠测试时消融区分训练收益。

单元测试：PYTHONPATH=.:tools python -m unittest test_birs。

## 8. 局限与下一步实验

- Top-N里没有正确候选时，重排无法修复；Top-5 Acc≈90%仅表示约90%的样本前五至少有一个正确框。
- 原Hungarian正候选不在集合中时，融合BCE仍监督其CRS logit；绑定排序项缺少集合内正例，仍跳过，不偷偷改标签或向Top-N注入GT。
- 正候选和完全不重叠负候选共存的样本可能有限，要观察有效监督数量。
- 绑定得分方向未必正确，不能仅靠增大β解决；参考物可能是背景，正确配置也未必满足文本。
- 无关系描述、多解标注、参考物漏检、原CRS过度自信仍是限制。
- 硬边界排序与跨集合概率数值未必一致，所有下游排序必须遵循priority接口。
- 本次只修正全候选融合BCE口径并删除旧BIRS兼容分支，没有增加门控、损失类型、外部监督或文本解析。推理仍有硬边界，BCE直接监督logit而非排序操作本身。性能是否提高需重新训练验证。

## 9. 本次验证范围

104项测试通过，覆盖全候选BCE数值及其梯度、正候选在Top-N外、没有绑定参考时的监督保留、Top-N硬边界、独立参考池及原PropVG/Grab/BCEM/correctness等回归路径。

GPU两样本测试验证联合/截断梯度下全256候选BCE；集合外正logit的梯度与全候选平均公式一致。两进程CPU DDP连续两步通过：两个rank都参与融合BCE，只有一个rank存在绑定配对，参数梯度仍同步。加入BIRS前BASE归档配置及真实检测头权重严格加载，初始化RNG和检测头推理输出与归档实现逐位一致。

本次未启动完整训练，未停止用户运行中的实验，也未修改既有实验归档。
