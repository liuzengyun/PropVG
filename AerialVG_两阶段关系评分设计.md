# 基于竞争感知共享参考证据的两阶段关系评分

**模块名称：** 两阶段关系细化（Two-Stage Relation Refinement，TSRR）。

**状态：** 首版代码、诊断接口与CPU验证已实现，正式性能尚待实验。不预设性能提升，不作首次提出的保证。

代码：`propvg/models/heads/two_stage_relation.py`；接入`REFHead.py`；训练仍使用`tools/train.py`。默认关闭，旧配置缺少新字段时仍运行原CRS。

## 1 研究动机、约束与创新边界

### 1.1 从初始匹配到关系消歧

航拍场景中同类别、同颜色的小目标往往密集重复。初始图文匹配可以排除明显不符的对象，但多个相似目标仍可能得到较高分数。参考物及其相对位置可以提供进一步消歧的证据。

原BEiT-3、检测Decoder和CRS已有隐式上下文交互，不能声称第一阶段严格只看外观。因此两个阶段准确称为“初始语义匹配”和“显式关系证据细化”。

### 1.2 关系成立，不等于证据有区分力

同一辆银色面包车可能同时位于多辆黑色轿车上方。每个目标独立寻找最有利参考物，可能让多个错误目标也得到高分。本模块固定同一个参考物，比较它对多个目标的支持，再考虑这些目标在CRS中是否真的互相竞争。

“与文本相关”“能区分候选”“实际关系正确”是不同概念。中间权重不等于真实参考身份或关系正确率，必须由最终目标监督和实验检查。

### 1.3 与BCEM的分工及原创性边界

BCEM增强候选的框外环境表示。本模块在CRS后输出标量修正，不回写候选特征、不新增框回归、不重跑编码器。当前用`relation_detach_inputs=True`隔离新增损失；设为`False`时才允许新增损失更新共享表示。“前向不回写”本身不等于“反向不影响”。

两阶段筛选与空间核验已有先例；AerialVG已有候选对矩阵及逐目标最大关系证据；G-Verifier已有几何验证重排。取消文本解析、改二维几何、加入softmax或方差，不是分别独立的强创新点。

当前待验证的贡献集中在：**面向重复布局，通过共享参考物的跨目标支持分配，以及目标竞争感知的证据权重，改善候选排序。** 需用相同候选池、相同评分器的聚合对照证实。

相关原文：[两阶段空间定位](https://arxiv.org/html/1707.05720)、[AerialVG](https://arxiv.org/html/2504.07836v4)、[G-Verifier](https://openreview.net/pdf/5f9b1c43a23d1be020c02bc7f2883c2e1e33b2ca.pdf)。

### 1.4 数据约束

不拆分文本、不提取关系词或三元组、不增加外部模型。新模块不读取teacher框、其他region标签、参考物GT。原始refer GT仅用于训练损失和离线分析。基线检测器可以继续使用原teacher设置，这不等于整套方法已是无teacher方法。

## 2 完整数据流与维度

```text
图像 + 完整文本
       ↓
BEiT-3：图像[B,768,36,36]、文本[B,64,768]（当前586输入）
       ↓
投影 + 原512通道共享SimFPN，检测侧四尺度各256通道
       ↓
Deformable DETR：256个候选特征、cxcywh框、检测二分类logits
       ↓
refer_adapter → 可选BCEM → 原CRS对全部目标候选评分
       ↓                              检测前景置信度
CRS概率Top-32                         ↓
       ↓                           Top-64，再高IoU去重，不补框
目标重复分组                         ↓
       └──────────共享参考池──────────┘
                       ↓
参考物读取完整文本 + 目标—参考6维几何 → 成对关系矩阵
                       ↓
沿目标组维度分配支持 → 竞争感知区分度
                       ↓
参考相关性/空参考选择 → 有界正负修正 → CRS短名单内部重排
                       ↓
保持原框及尾部顺序，输出最终Top-1
```

记原候选数N=256，目标预算K=32，参考预算64，实际参考数R≤64，目标组数G≤K，投影维度D=128。

| 数据 | 维度 | 操作目的 |
|---|---|---|
| 候选/框/检测logits | [B,N,256] / [B,N,4] / [B,N,2] | 保留原检测输出 |
| 完整文本投影 | [B,64,128] | 全token输入，屏蔽padding |
| 参考文本交互结果 | [B,R,128] | 每个参考只做一次文本注意力 |
| 成对几何/交互 | [B,G,R,128] | 绑定参考语义与实际位置 |
| 关系logit/支持分布 | [B,G,R] | 同一个参考跨目标比较 |
| 可微最终logit | [B,K] | 新增最终排序监督 |
| 对外分数/priority | [B,N,1] | 保留slot、尾部并兼容原后处理 |

## 3 坐标与两个独立候选池

### 3.1 有效图像坐标

框停止梯度，按原后处理顺序执行：cxcywh转xyxy、转float32、乘有效图像宽高、clip到有效图像、再归一化。退化框与非有限框不参与选择。padding不进入物体关系几何，不能用pad尺寸代替有效图像尺寸。

**目的：** 与正式评估一致，避免直接挪框迎合关系。仅在关闭输入截断时，共享特征才接受新增损失，因而可以间接影响检测结果。

### 3.2 目标池与重复分组

使用原CRS概率和CPU正式评估的并列规则取有效Top-32。按CRS顺序贪心建立代表框：IoU≥0.8归入首个匹配组，否则新建一组。不是传递闭包聚类。组级关系修正广播给组内框，原框与原CRS分数保留。

**目的：** 同一辆车的多个框不应被当成多个独立竞争物体。高IoU是近似启发式，不保证严格识别物理实例。

### 3.3 参考池先检测Top-64，再去重

前景分数取检测二分类softmax第0类；按置信度稳定降序取64，再去除与已保留框IoU≥0.8的框。不补框，因此R≤64。所有目标共享参考池，不按目标CRS分数筛参考，也不各选最近6个邻居。

仅屏蔽目标与参考的同实例高重叠配对。不要求IoU=0，真实的部分重叠、包含关系仍可使用。

基线test中检测Top-64的**目标**召回94.7914%、相对256框的目标保留99.3344%；CRS Top-32命中94.6644%。这是特定基线去重前的目标统计，不是参考物召回率，也不保证新模型具有相同上限。超参数应在val确定。

**Grab组合：** 目标仍来自Grab后的CRS池；参考独立来自原始未过滤DETR候选，经过同一refer_adapter，不再被Grab目标过滤删除。无Grab时参考复用可选BCEM后的CRS输入；有Grab时未过滤参考不额外跑BCEM/Grab。这是明确不同的参考特征路径，应记录配置和原DETR索引。

## 4 文本引导的参考—几何评分

### 4.1 六维几何

$$
g_{ij}=[\Delta x,\Delta y,\sqrt{\Delta x^2+\Delta y^2},
\log((w_j+\epsilon)/(w_i+\epsilon)),
\log((h_j+\epsilon)/(h_i+\epsilon)),\operatorname{IoU}(b_i,b_j)].
$$

偏移按有效图像归一化；log比例限制到[-10,10]防止异常数值；MLP输出128维几何表示φ。上方通常对应负Δy，但“in front of”不能硬编码为图像上方。

**目的：** 提供方向、距离、尺度和重叠事实，不另行预测关系类别，不需要关系标签。

### 4.2 完整文本读取与乘性交互

$$
q_j=P_v(C_j),\quad t=P_t(T),\quad z_j=\operatorname{Attention}(q_j,t,t),
$$

$$
h_j=\operatorname{LN}(\operatorname{GELU}(W[q_j;z_j;q_j\odot z_j])),
\qquad e_{ij}=w^\top(h_j\odot\phi(g_{ij})).
$$

注意力4头、128维、dropout=0。每个参考的文本交互复用于全部目标，不为每个候选对重复运行Transformer。padding严格屏蔽，全padding时所有关系无效。

**目的：** 评分前绑定参考语义与几何，减少独立汇总。关系评分不直接输入目标外观C_i，但参考特征已有图文上下文，不能声称严格消除了外观捷径。

## 5 支持分配、竞争感知与空参考

### 5.1 固定参考，沿目标组归一化

令S_j为参考j的合法目标组，M_j为组数：

$$
v_{ij}=\operatorname{softmax}_{i\in\mathcal S_j}(e_{ij}/\tau_r).
$$

**目的：** 比较同一个参考究竟支持谁，不让每个目标独立寻找最有利邻居。不施加一对一匹配；全无效列严格为0，不能产生NaN。

### 5.2 竞争先验与低分候选底座

$$
p_{ij}=(1-\rho)\operatorname{softmax}_{i\in\mathcal S_j}
(\operatorname{stopgrad}(\ell_i^{CRS})/\tau_a)+\rho/M_j.
$$

默认ρ=0.2。**目的：** 关注当前仍在竞争的目标，又不让CRS自信选错时完全排除正确低分候选。此处统计先验始终停止梯度；残差相加的实际CRS logit是否接受新增损失则由`detach_inputs`控制。p不是校准正确率。

### 5.3 不采用原草案的全候选熵

$$
\bar v_j=\sum_i p_{ij}v_{ij},\qquad
d_j=4\sum_i p_{ij}(v_{ij}-\bar v_j)^2\in[0,1],
\qquad \widetilde d_j=\delta+(1-\delta)d_j.
$$

默认δ=0.1。**目的：** 衡量能否区分真正竞争的目标；保底权重避免均匀初始化关闭学习信号。排除许多低分背景，不应直接等同于能区分高分A、B。这仍是待验证假设，不是关系正确性判断，也不额外加入低熵损失。

### 5.4 参考相关性与空参考

$$
u_j=W_a h_j+b_a,\quad
u_\varnothing=W_\varnothing\operatorname{Mean}_{valid}(t)+b_\varnothing,
$$

$$
[a_\varnothing,a_1,\ldots,a_R]=\operatorname{softmax}([u_\varnothing,u_1,\ldots,u_R]).
$$

没有合法配对的参考权重为0。**目的：** 无参考描述、参考缺失或证据不可靠时允许退回CRS。所有描述走同一个网络，不解析关系词，不需要无参考标签。空参考权重不是无参考识别准确率。

## 6 有界修正、正式排序与退化行为

$$
r_i=\sum_{j:i\in\mathcal S_j}a_j\widetilde d_j(M_jv_{ij}-1),\qquad
\Delta_i=\lambda\tanh(r_i),
$$

$$
\ell_i^{final}=\ell_i^{CRS}+\Delta_i,\qquad
\lambda=\lambda_{max}\sigma(\theta_\lambda).
$$

默认lambda上限2、初值0.1。实现用M_j(v_ij−1/M_j)减少均匀票的舍入误差。

每步目的：均匀支持产生0修正；相对支持强则加分、弱则减分；乘M缓解候选数导致的稀释；tanh及受限lambda防止初期大扰动；非零初始化保留学习通路。

全无效框、全padding文本、没有合法配对、仅一个目标组时修正为0，排序保持原CRS（包括概率并列规则）。仅重排最初Top-K，尾部保持原顺序，不让第K+1框因其他框降分插入短名单。

对外`ranking_priority`是唯一顺序编码，**不是概率或实际logit**；实际CRS/final logit另行诊断保存。不能绕过priority单独用sigmoid排序。

非零修正采用真实logit排序，可能涉及旧概率并列的精度差异。分析额外保存`crs_logit`：在同一短名单仅按原logit重排，用于识别精度效应，不将其全部归因于关系推理。

## 7 同步训练、唯一新增损失与可选梯度截断

$$
\mathcal L=\mathcal L_{PropVG}+\mu\mathcal L_{final},\qquad
\mathcal L_{final}=-\log
\frac{\sum_{i\in\mathcal P}\exp(\ell_i^{final})}
{\sum_{i\in\mathcal S}\exp(\ell_i^{final})}.
$$

原检测、原CRS定位和其他已开启模块损失不变，仅增加`loss_relation_final`，默认μ=1。正集P为已选目标框中与原始refer GT的IoU≥0.5者；多正例概率质量监督避免将多个正确框互相作为可靠负例。IoU标签计算停止梯度。

短名单无正例时跳过该样本的新损失，原损失照常训练；不插入GT框、不用GT补候选。没有refer目标的样本不计算新增损失，这与“没有参考物描述但仍有目标”不同，后者正常执行全部流程。

DDP按全局有效样本均值归一化：局部loss和乘world_size再除全局有效数。全卡无正例时返回图连接零，不除零。

`relation_detach_inputs`控制新增损失是否影响第一阶段，当前配置设为`True`：

- `True`：候选/参考特征、文本特征、CRS概率及残差相加使用的CRS logit均先`detach()`。新增损失只更新第二阶段参数；原CRS、BCEM及主干仍由各自原有损失正常训练，并非冻结或分阶段训练。
- `False`：保留之前的联合梯度，新增损失可以回传到CRS及共享表示。

截断时的实际监督为`final_logit = stopgrad(CRS_logit) + relation_delta`。**目的：** 保留第一阶段的正向分数作为修正基准，但阻止新增监督损伤原定位能力。单独截断特征还不够，直接残差上的CRS logit也必须截断。

截断开启时，第二阶段与原模型分别做梯度范数裁剪。**目的：** 避免新增梯度进入同一个全局范数，间接改变原CRS的更新缩放。截断关闭时仍按原联合方式裁剪。

| 路径 | 新增损失反传 | 目的/说明 |
|---|---|---|
| 关系评分、参考权重、lambda | 是 | 学习有效修正 |
| 实际CRS分数及参数 | 仅`detach_inputs=False`时 | 原定位监督始终保留 |
| 候选/文本、共享Decoder/FPN/BEiT-3、BCEM | 仅`detach_inputs=False`且未冻结时 | 截断开启时只接受原有损失 |
| 竞争统计中的CRS先验 | 否 | 不直接操纵统计分布 |
| 框坐标、检测置信度筛选 | 否 | 不直接挪框迎合关系 |
| Top-K、去重、分组、priority | 否 | 离散布局与评估顺序 |

原CRS及历史文本池化行为未在本任务修改，不暗中改变基线。

### 7.1 初始化、checkpoint与组合

关闭/缺省不构建模块，不新增参数或消耗新RNG。开启时在旧模块初始化后，以局部CPU RNG初始化并恢复随机流，保持旧参数初值。

有有效新增监督的optimizer step更新`trained_steps`；尚无此类step时验证退回CRS，避免随机关系头参与早期验证。加载检查新模块完整权重、schema和策略签名，防止Top-K、温度、去重等无形状变化被静默修改。

旧基线增加新头用`--finetune-from`；`--load-from/--resume-from`要求完整兼容新模型。旧模型在新开关关闭时仍可测试。与BCEM/Grab/正确性头可组合；与旧CRV同时开启会明确报错，避免未经设计的串联重排。

梯度截断是纯训练行为，不增加权重或buffer，不改变推理数值，也不进入checkpoint策略签名。以前训练的第二阶段权重仍能严格加载；旧配置未设置`detach_inputs`时默认`False`，保留之前的联合梯度逻辑。恢复训练时修改此开关会改变训练方法，需保留对应归档配置。

## 8 配置、运行与可控消融

`configs/aerialvg/PropVG-aerialvg.py`：

```python
enable_relation_refinement = False  # 改True开启，保留原CRS
relation_detach_inputs = True  # 新增损失不回传第一阶段；False恢复联合梯度
relation_target_topk = 32
relation_reference_topk = 64
relation_refinement = dict(
    target_topk=relation_target_topk, reference_topk=relation_reference_topk,
    detach_inputs=relation_detach_inputs,
    duplicate_iou=0.8, model_dim=128, num_heads=4,
    prior_temperature=1.0, relation_temperature=1.0, prior_uniform_mix=0.2,
    discriminability_floor=0.1, initial_strength=0.1, max_strength=2.0,
    loss_weight=1.0, aggregation='competitive',
    use_geometry=True, use_reference_semantics=True,
)
```

Top-K控制预算；duplicate_iou控制重复近似；temperature控制相对分布尺度；prior_uniform_mix保护低分候选；discriminability_floor保留初始梯度；strength设置控制修正幅度；loss_weight是唯一新增损失权重。

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/train.py \
  configs/aerialvg/PropVG-aerialvg.py \
  --work-dir work_dir/aerialvg/PropVG-TwoStageRelation \
  --cfg-options model.head.enable_relation_refinement=True

# 可选：追加 --finetune-from /原基线/top1_best.pth
# 同一训练循环，不冻结主干、不使用缓存；新增损失是否反传第一阶段由detach_inputs控制。

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tools/eval_aerialvg.py \
  /本次实验/归档配置.py --checkpoint /本次实验/top1_best.pth --split test
```

原dist_train.sh读取同一个配置，改True后可原样多卡训练。CLI覆盖需修改展开后的`model.head.*`，只覆盖顶层Python变量不会重新展开引用。例如`--cfg-options model.head.enable_relation_refinement=True model.head.relation_refinement.detach_inputs=True`。正式评估使用训练归档配置。

聚合消融：`competitive`完整版本；`vote`跨目标投票、不加区分度；`weighted`参考权重汇总tanh成对分数；`max`取参考权重乘tanh成对分数的最大值；`crs_control`修正为零但保留同样新增最终监督。保持候选池与评分器一致，仍需分别训练。

`use_geometry=False`去几何；`use_reference_semantics=False`为几何主导对照。消融改变策略签名，测试使用各自归档配置；推理时临时改设置不等于正式重训消融。

## 9 训练日志与可验证接口

控制台与原SwanLab增加`loss_relation_final`及`relation_refinement_*`，正常训练不导出完整矩阵。

| 字段 | 含义 |
|---|---|
| lambda | 当前修正尺度，不是准确率 |
| null_weight | 空参考权重，不是无参考正确率 |
| reference_count / group_count | 去重参考数/目标组数 |
| discriminability | 支持区分度，不是可靠性 |
| correction_abs | 平均绝对logit修正 |
| rank_changed | 同一前向的Top-1改变比例 |
| reachable | 短名单有IoU≥0.5框的比例 |
| corrected / harmed | 相比本模型第一阶段的改对/改错比例 |
| supervised_samples | 全卡新增loss有效样本数 |
| ready | 训练路径状态，不代表已学会关系 |

比例使用0–1。训练日志为迭代均值，完整数据的精确计数以离线脚本为准。

### 9.1 一次主干前向完成分析

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python tools/analyze_two_stage_relation.py \
  /本次实验/归档配置.py --checkpoint /本次实验/top1_best.pth \
  --split test --batch-size 4 --workers 4 \
  --output-dir /本次实验/two_stage_analysis_test

# 小规模检查追加 --max-samples 100
# 独立原基线比较追加：
# --baseline-report /原基线/detector_topk_recall_test_20261103/per_sample.csv
```

脚本推理前移除GT，再在外部算指标。主干每批只前向一次，通过`with head.relation_refiner.diagnostics():`获取`extra_dict['relation_refinement_samples']`，退出context恢复开关。诊断不改变注意力内核路径，不永久持有GPU特征。

输出文件：

- `summary.json`/`report.md`：CRS、CRS-logit、relation-only、final四种排序的Top-1/Top-5/mIoU、Top-K、改对/改错/净变化、池覆盖、空参考与修正幅度、按CRS熵分桶的纠错计数。
- `per_sample.csv`：索引、文件名、完整描述、四种排序的Top-1索引/IoU、Top-K命中、修正幅度、候选数。
- `details/000000.npz`：有效图像归一化框、参考框、Top-64去重前后索引、原DETR映射、分组、6维几何、配对mask、关系logit、支持分布、竞争先验、参考权重、区分度、文本注意力、CRS/final logit、修正及完整排序。外部GT另外标明。

索引含义：`target_indices`是CRS输入池slot；`target_source_indices`映射原DETR；`reference_indices`指参考来源池（有Grab时仍为原DETR）；`group_slots`是Top-K内部代表位置。padding条目结合对应valid mask读取。

relation-only仅按修正排序，是诊断而非完整独立定位模型；crs_logit识别精度/并列效应。给定独立基线CSV时，严格匹配索引、文件名、描述，报告“原基线→本模型CRS”“原基线→final”，避免只与已经退化的第一阶段比较。特别是未截断训练时，不能把同一模型的第一阶段默认当作独立基线。

## 10 潜在局限、实验与验证范围

重点验证：最终超过独立基线；改对大于改错；同步训练未损伤CRS；参考语义和几何确实同时使用；额外监督本身与关系机制的收益分开。开启梯度截断隔离了新增损失，但不承诺额外重排一定提升性能。

至少比较原CRS、crs_control、max、weighted、vote、competitive，以及有无BCEM。保持teacher数量、数据划分、图像尺寸、检测损失、初始化和随机种子一致；不在test反复选超参数后报告无偏提升。

具体注意：当前配置的teacher_max10=True，而20261015_104446归档配置没有该字段、对应默认False。本次未改动用户已有数据设置。旧基线CSV可辅助诊断，但严格新对照应使用同一份配置只切新模块开关，不能把不同teacher过滤设置的差异直接归因于模块。

已知局限：CRS错误先验仍会影响区分度（均匀底座仅缓解）；空参考/lambda可能关闭分支；高置信背景可能入池；高IoU去重并非实例识别；道路/标线等参考可能没有框；二维几何不覆盖全部三维语言；加权和不保证多个关系严格同时成立；Top-K有召回上限；单目标监督不能唯一识别真实参考物。注意力/权重不能直接作为因果解释。

CPU检查入口：

```bash
PYTHONWARNINGS=ignore PYTHONPATH=.:tools python -m unittest discover -s tools -p 'test_two_stage_relation.py'
```

覆盖选择顺序、去重不补框、目标分组、真实重叠、padding、支持归一化、均匀/空参考、梯度和GT边界、DDP全局均值、完整head、关闭兼容、checkpoint策略、诊断不扰动与离线改对改错统计。CPU检查不能替代真实GPU/多卡训练与正式性能实验。

### 10.1 本次验证记录

2026-11-05：新模块33项专项检查及相关回归合计212项CPU测试通过；171个Python文件语法检查通过。另与修改前REFHead快照逐位对照base、Grab、mask、BCEM、correctness、CRV六种配置，关闭新模块时初始化、RNG、训练loss、参数梯度与推理输出一致。

同日增加梯度截断开关后：专项检查增至36项，相关CPU回归合计215项通过；171个Python文件语法检查及`git diff --check`通过。新检查覆盖：输入/CRS logit不接收新增梯度、第二阶段自身仍能学习、截断前后前向逐位相同及严格加载兼容；在BCEM关闭/开启两种情况下，原模型梯度、独立裁剪后的梯度以及一次Adam更新均与未添加第二阶段的对照逐位一致。

相关回归不包含已有CRS整数文本mask池化的5个测试方法（含7个失败子项）；它们在修改前已复现，本次保留原实现，未将其修复混入新模块。执行环境当前不能连接NVIDIA驱动，因此未进行正式GPU/多卡训练、全量新模型评估或性能结论验证。
