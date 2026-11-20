# 航拍视觉指代的文本条件竞争集合与关系绑定细化（BIRS）

> 2026-11-20清理：旧独立 `relation_refinement` 模块及其专属实现、评估、分析脚本、测试和设计文档已移除，
> CRV实现、专用聚类辅助文件、配置、训练/推理接入、校验和文档也已移除。
> correctness观察预测头及其训练/评估接入、配置、专属测试和文档也已移除；普通定位指标和错误分析保留。
> BIRS v1–v5、BCEM、CRS及检测Top-K开关保留。旧归档配置里的关闭字段仍可接收，但不构建任何分支；
> 设置旧 `enable_relation_refinement=True` 会明确报已移除。历史测试记录中的该模块仅指当时的版本，不代表现可运行；
> 实验目录及其代码备份未删除。旧模块若需恢复可从Git历史取回。
> `enable_crv=False`及旧`crv`字典仅作兼容接收；设为True会明确报已移除。历史CRV测试记录仅代表当时状态。
> `enable_correctness=False`及旧`correctness`字典同样兼容接收，不构建任何参数；设为True明确报已移除。
> 移除correctness后180项CPU回归通过，包含原关闭配置的参数/RNG/逐值前向兼容、BIRS/BCEM/Grab、
> 检测Top-K和实验追踪；训练内验证与独立评估均保持Top1/Top5/mIoU，且GT不传入模型。未启动正式训练。
> 两个旧模块清理后191项CPU回归通过，包含旧关闭配置的初始化/RNG、严格权重加载及逐值前向兼容检查，
> 以及BIRS、BCEM、Grab、correctness、检测Top-K和实验追踪回归；未启动正式训练。

## 第五版 v5：候选专属近邻参考的关系竞争细化

更新日期：2026-11-20。实现：`propvg/models/heads/binding_residual_v5.py`，类名
`LocalBindingResidualScoring`。本版独立接入，不修改 v3/v4 源文件；下文保留历史版本说明。
配置通过 `enable_birs=True`、`birs_version="v5"` 启用。为避免影响原实验，主配置当前仍默认
`"v3"`，且不自动改变用户设置的 BCEM、Grab、teacher、工作目录等选项。

### 1. 动机与修改边界

同一张航拍图中的不同候选，通常处于不同的局部物体配置中。全图前景池虽然覆盖较广，但也把远处、
与当前候选关系较弱的物体一起送入关系评分。v5 先为各目标假设建立独立的局部参考集，再利用完整文本
判断哪些邻居有用，使关系验证围绕该候选附近的场景展开。

这不是新增一个目标检测器，也不解析“黑车”“左边”等文本成分。不需要关系标注、参考物 GT 或新外部模型。
仍保留原 CRS 初始评分，以及 v3 的聚类自适应连续 Top-N、关系支持差、可学习残差和完整候选 BCE。
文本交互、sparsemax 和 null 本来就存在于 v3；本次核心变化是**候选专属、最多8个的零重叠近邻池**。
本版是待实验验证的结构改动，不保证性能提升，也不声称实现了严格因果识别。

### 2. 每一步的做法与目的

#### 2.1 保留 v3 的目标竞争集合

对原候选特征做8类确定性聚类，依据 CRS 前3名所属的簇，在前32名中找最远覆盖位置，得到至少5、
最多32个的连续排名前缀。所有中间排名都纳入，不仅保留某个簇。目的：保持原实验的可比较性，
仅对已有较高响应的目标假设做关系细化。

#### 2.2 先找可靠的前景参考池

使用原检测器前景置信度，从原解码器候选中**先取 Top-64，再按置信度以 IoU≥0.8 去重**，不补满。
参考特征仍来自 pre-BCEM 的原解码器，经原 `refer_adapter`；不因为 Grab 裁剪了目标集合而丢掉原参考池。
目的：不直接把256个候选中的低质量背景框都当作参考。Top-64仅是前景置信度排序上限，并非绝对置信度保证。

#### 2.3 每个候选分别选最近8个合法邻居

对于目标假设 i：

1. 排除与自己源 query ID 相同的参考。
2. 排除与自己有任意正面积交集的参考，要求 `IoU(target, reference)==0`。边界相接、交集面积为0的框允许保留。
3. 按候选中心与参考中心的欧氏距离排序，选最近至多 `neighbour_topk=8` 个。距离相同保留前景池原顺序。
4. 不足8个不补重复框，空槽严格掩码；每个候选的参考集独立计算，允许不同候选共享某个真实参考。

框先按原有效图像宽高归一化。计算距离时再乘 `[W/min(H,W), H/min(H,W)]`，恢复等比例距离：

$$d_{ij}=\sqrt{\left(\frac{x_j-x_i}{\min(H,W)}\right)^2+
                      \left(\frac{y_j-y_i}{\min(H,W)}\right)^2},$$

上式 x、y 为有效图像中的像素中心坐标；实现不依赖 padding 大小，不把非方图横向和纵向距离错误地等权缩放。
目的：邻居代表当前候选附近的独立物体，而不是同一目标的另一个框、局部框或包含目标的大框。

“零重叠”指**参考与当前候选**之间；参考彼此只沿用前景池的近重复去重，未强制所有参考之间 IoU=0。
这避免改变之前约定的 Top-64 去重策略，也不把真实不同但部分重叠的物体全部删除。

#### 2.4 文本筛选：最多8个，不是必须使用8个

沿用 v3 的共享关系评分器。目标、参考分别投影到 D=128；相对位置、方向、距离、宽高比、
目标尺度归一化偏移组成9维几何，并投影到128维。对每个邻居构造：

$$e_{ij}=\operatorname{MLP}([q_i;v_j;\phi(g_{ij});q_i\odot v_j]).$$

这些关系 token 读取完整文本 T，再得到每个参考的 evidence logit；目标和句子表示共同产生 null logit：

$$a_i=\operatorname{sparsemax}([\ell_{i1},\ldots,\ell_{iH},\ell_{i\varnothing}]),
\qquad H\le8.$$

非法槽位永远权重0，null始终合法且 value=0。无额外软注意力保底、无强制参考最小权重。
稀疏选择后的有效参考数量可以为0至8；它是模型选中的证据数量，不等于真实参考物数量。
本版采用稳定的 masked sparsemax，避免非法槽位的有限负占位数超过极低的合法 logit。

汇聚关系后，沿用 v3 的目标＋句子＋关系汇聚 support MLP，得到未校准的标量支持分，**不是概率**。
目的：几何近邻只负责提供局部证据，文本决定哪些邻居对这句描述有用，不手工拆解描述。

#### 2.5 固定局部集合做反事实对照

事实分支用真实语义—几何对应。对照分支固定同一目标、同一文本、同一参考语义集合及其数量，
只在该候选合法的局部参考内部做两次确定性非原样循环置换，交换完整9维几何向量：

$$S_i^+=F(q_i,\{(v_j,g_{ij})\},T),\qquad
S_{i,m}^-=F(q_i,\{(v_j,g_{i\pi_m(j)})\},T).$$

沿用 v3 的完整几何重绑定，**不是 v4 的中心交换/尺寸保持策略**。不移动图像像素、不修改检测框，
不重新搜索邻居，不让 null 参与置换，不引入 GT。每个分支共享评分器并重新计算文本注意力，
允许破坏对应后参考的相关性发生改变。最多8个关系 token 真正进入评分，而非对64个全部计算后才掩码。
目的：比较当前候选得到的支持是否来自语义与空间配置的正确结合，而不仅是附近物体的存在。

#### 2.6 无参考就不用关系残差

若合法邻居不足两个，无法构造置换对照，令关系差为0。若事实路径完全选择null，也令关系差为0，
即使某个对照路径重新分配注意力给真实参考，也不据此强加负残差。

$$\Delta_i=\begin{cases}
S_i^+-\frac1M\sum_m S_{i,m}^-, &\text{至少两个合法邻居且事实路径读取了真实参考},\\
0,&\text{否则}.
\end{cases}$$

$$z_i^{\mathrm{final}}=z_i^{\mathrm{CRS}}+\beta\Delta_i,
\qquad\beta=2\operatorname{sigmoid}(a),\quad\beta_{\mathrm{init}}=0.1.$$

不使用熵门控、不添加新融合网络；仅在原连续 Top-N 内输出排序。新分支未进行任何优化步时，
测试仍回退原 CRS 分数。注意：回退分数不等于取消原有 Top-N 输出限制。

### 3. 整体数据流与维度

设 B 为 batch，P 通常256个原检测候选，N 为每图自适应竞争数，K 为本 batch 最大 N，
R=min(64,P) 为带掩码的全局参考池宽度，H=min(8,R) 为局部带掩码宽度，L 为文本长度。

```text
BEiT-3 + SimFPN + 原检测解码器
  ├─ 目标候选特征（可经原Grab/BCEM）[B,P,256] → 原CRS logits[B,P,1]
  │     └─ v3聚类/排序选择 → 目标[B,K,128]，目标框[B,K,4]
  └─ pre-BCEM原检测特征 + 检测前景得分
        └─ Top64再去重 → 参考池[B,R,128]，框[B,R,4]，有效mask[B,R]
               ↓ 每目标：排除自身/正面积重叠，按等比例中心距离取最近8个
        局部参考[B,K,H,128]、原query索引[B,K,H]、合法mask[B,K,H]
               ↓ 目标 + 局部参考 + 9维几何 + 逐元素交互
        关系token[B,K,H,128] ← 完整文本[B,L,128]
               ↓ 文本交叉注意力 + sparsemax(局部参考/null)
        事实support[B,K]、注意力[B,K,H+1]
               ↓ 固定局部集合、两次几何循环重绑定，共享评分器
        对照support[B,K,2]
               ↓ 有参考时计算差值，否则0
        Δ[B,K] → scatter至原候选[B,P] → 原CRS logit + βΔ
               ↓ 仅在Top-N内排序，返回目标框；检测框本身不变
```

单个目标的文本注意力实际 query 数最多8。为兼容已有分析接口，保存的 `selected_attention`
会将局部权重 scatter 回 `[B,K,R+1]`，未入近邻的全局参考权重为0，最后一位仍为null。

### 4. 损失、梯度和训练/测试一致性

$$\mathcal L=\mathcal L_{\mathrm{det}}+\mathcal L_{\mathrm{other}}
              +1.0\mathcal L_{\mathrm{CRS\ BCE}}+0.5\mathcal L_{\mathrm{final\ BCE}}.$$

两项BCE保持 v3 的原 Hungarian refer 标签、全有效候选监督、每图/全局有效样本归一化。
不只监督Top-N，未进入Top-N的候选也保留BCE。原 head 的 refer 权重仍乘两项BCE；其余检测、
Grab、BCEM 等损失不变。本版不加独立CF hinge，不采用v4的软参考路径。

`detach_inputs=False`默认联合优化：最终BCE可影响CRS、关系评分器、融合强度和候选/文本上游。
框坐标、前景排序、去重、近邻和几何置换均为离散/截断路径，不通过该模块新增框坐标梯度。
`detach_inputs=True`保留 v3 行为：新增分支截断输入，原CRS BCE仍训练原模型。
空GT/空参考/全padding时保持有限损失和连通零梯度，兼容DDP无效样本。

训练和测试使用同一前景池、近邻筛选、文本读取和确定性对照；只有loss读取GT，不因测试结果改变阈值。

### 5. 使用、旧版本兼容和验证接口

主配置 `configs/aerialvg/PropVG-aerialvg.py` 中：

```python
enable_birs = True
birs_version = "v5"  # 原默认仍为v3，手动改这一项启用。
# v5配置分支中：
# neighbour_topk=8, reference_topk=64, duplicate_iou=0.8,
# final_mix=0.5, detach_inputs=False, checkpoint_chunks=True
```

建议自行给v5设置独立 `work_dir`；原配置当前 BCEM 开关保持用户已有值，切版本不会自动开启BCEM。
若对照组合实验，须自行保持两组 `enable_bcem` 等其余配置一致。

v5 schema=6，近邻数/距离定义/零重叠规则/对照/null策略进入独立策略签名。
v3/v4权重不能当v5直接测试或断点恢复；要初始化新分支，使用原有 `--finetune-from` 工作流。
v3评估仍使用v3归档配置；`enable_birs=False` 仍不构建分支，旧baseline/Grab/BCEM模型加载规则不变。
本次没有更改 `binding_residual_v3.py`、`binding_residual_v4.py` 或通用checkpoint加载器。

通过 `with model.head.birs.diagnostics():` 和 `extra_dict['birs_samples']` 可检查：

- `target_indices[B,K]` / `target_valid`：目标假设的原候选索引和有效性。
- `local_reference_indices[B,K,H]`：原检测query索引；无效槽为−1，必须结合 `local_reference_valid`。
- `local_reference_slots[B,K,H]`：在共享Top-64池内的槽位，用于对齐权重。
- `local_reference_distances[B,K,H]`：短边归一化中心距离；无效槽存0，不可把它当真实最近距离。
- `reference_boxes` / `reference_confidence`：全局前景池的框与检测置信度。
- `selected_attention[B,K,R+1]` / `selected_control_attention[B,K,2,R+1]`：事实/对照文本权重。
- `selected_support[B,K]` / `selected_controls[B,K,2]` / `delta[B,P]` / `eligible[B,P]`：支持分、最终残差及适用mask。

`null` 这个历史诊断字段依旧表示**对照support均值**，不是null注意力！null权重须读取
`selected_attention[..., -1]`。`reference_count`仍统计共享池；局部数量读取
`local_reference_valid.sum(-1)`，文本后有效数量读取 `selected_sparse_reference_count`。

专项测试：`PYTHONPATH=.:tools python -W ignore -m unittest test_birs_v5`。
测试覆盖等比例近邻距离、严格零重叠、变长/空参考、固定对照池、null回退、全候选BCE权重、
梯度边界、checkpoint重算、策略签名、head训练/测试和原模型初始化隔离；旧v3归档数值回归另外保留。
这些是CPU小型逻辑验证，不代表完整GPU训练或性能验证。

2026-11-20验证记录：196项CPU回归通过，包含v5专项13项，以及v1/v2/v3/v4、BCEM、Grab、
correctness、可选管线和原两阶段关系模块；其中保留直接加载指定v3实验归档源码的逐值数值对照。
未启动正式数据训练，也未运行CUDA或多进程DDP训练；不能用本记录声称性能改善。

### 6. 待验证的局限与建议实验

最近8个可能漏掉远处真正参考；严格零重叠会排除真实重叠物体；检测Top-64不保证参考都是前景。
局部邻居外观相近时，重绑定可能只是近似等价的排列；几何破坏也不保证违反这句文本。
null全选时关系分支没有有效残差，可能发生学习不足；本版不为了防塌缩强制读取参考。
离散近邻选择无梯度，但关系特征仍参与联合训练，不能把“离散筛选截断”误解为整个模块冻结。

建议与相同训练设置的v3比较：Top1/Top5/mIoU、每候选局部数量、文本筛选后数量、null权重、
关系残差、改对/改错数和近邻可视化。若退化，先区分“近邻漏参考”“null垄断”“反事实无区分度”，
不要直接叠加额外损失。近邻数4/8/12只能在val消融，不能用test选超参数。

---

## 公共可选开关：检测 Top-K 后再做 BCEM / CRS / BIRS

更新日期：2026-11-20。主配置新增：

```python
enable_detector_topk = False  # True启用；False或旧配置缺省保持原流程。
detector_topk = 64
```

开关独立于 `enable_birs` 和 `birs_version`。`num_queries=256` 不变，不是把检测器改成64个query。
开启时的数据流：

```text
检测解码器256个query → 检测头256个框/前景logit
  ├─ 全部256个 → 原多层检测损失 / Hungarian匹配（不变）
  └─ （若有Grab，先保留其原处理）
       → 按检测前景概率取至多64个，特征/框/类别/原query ID同步gather
       → BCEM（若开启）→ CRS → 所选版本BIRS（若开启）→ 最终目标框
```

前景置信度是检测二分类 `softmax(class_logits)[...,0]`，不是原始单通道logit，不是CRS分数。
排序稳定、无GT、无文本依赖；不做额外NMS，也不强行把refer GT补进集合。训练和测试均用同一筛选。
如果开启Grab，是在它保留的候选中按检测置信度取Top-K，不恢复已被Grab删掉的query；不开Grab时即原256个中的检测Top-K。

BIRS的参考仍从原pre-BCEM检测池取，保留原query ID；其前景参考池上限自动取
`min(birs.reference_topk, detector_topk)`，再执行原去重/v5近邻筛选。因此默认K=64时，
BCEM/CRS目标数最多64、BIRS目标来自这64个，BIRS参考也最多为原检测Top-64，不能理解为只保留CRS Top-64。
参考投影依旧能读取原解码器，但关系评分不会把第65名之后的框当参考。

训练标签先在256个检测query上完成原Hungarian匹配，然后按最终保留的原query ID同步gather，
不重新匹配或改变GT。BIRS v3/v4/v5沿用原有效样本规则：匹配refer正query不在保留集合时，
该图没有有效正候选，两项定位BCE不计入该图；256个query的检测损失继续训练。仅CRS/旧其他分支
保留各自原损失逻辑，可能会对该图保留的全负候选计算BCE。本开关没有额外修改各模块的监督定义或权重。
Grab的keep惩罚只判断Grab自身是否删除GT，不因之后的检测Top-K删除而错误惩罚Grab过滤器。

没有新增参数或buffer，默认关闭不消费随机数、不更改state_dict。旧模型仍使用原配置测试；
旧权重用新开启配置推理会改变CRS的候选自注意力上下文和候选覆盖，不能声称与原评估等价。
若BIRS参考上限因K改变，原有策略签名校验也可能拒绝直接加载不匹配的任务checkpoint。

分析接口：`extra_dict['grounding_source_indices'][B,Q_kept]` 保存当前位置对应的原解码器query ID。
开启本开关后，BIRS的 `target_indices` 是缩减后目标集合中的槽位；用
`grounding_source_indices.gather(1, target_indices)` 可恢复原解码器ID。
训练阶段另保存 `grounding_refer_exists` 和 `grounding_refer_retained` 供检查筛选召回。
只做分析，不作为筛选条件或额外输入，不增加默认日志输出。

建议先关注检测Top-64的refer召回：该步骤可能丢失真实目标，且早期检测分数未收敛时更明显。
减少后续BCEM/CRS的候选计算量不等于缩减检测器或BEiT-3计算量，整体加速幅度需实测。
专项CPU测试：`PYTHONPATH=.:tools python -W ignore -m unittest test_grounding_detector_topk`。
本次203项CPU回归全部通过（其中本开关专项7项），包含原各版本BIRS、BCEM、Grab、correctness、
可选管线和两阶段关系模块；验证了关闭开关的参数/RNG兼容、完整检测监督、标签索引对齐和Grab惩罚边界。
未启动正式训练或GPU性能评估。

---

## 历史方案：第四版 v4——竞争式反事实关系细化

更新日期：2026-11-18。代码：`propvg/models/heads/binding_residual_v4.py`。
主配置选择 `birs_version="v4"`，`enable_birs=False` 时完全不调用该模块。
v1/v2继续保留；`birs_version="v3"` 对齐用户指定的`PropVG-birs3+bcem8/20261118_084936`实验归档，而非更早Git提交。该实验已将CRS BCE改为1，融合BCE为0.5，无CF损失、无软注意力混合。
v3与v4的两项BCE均为1/0.5；v4另加0.1倍CF损失。公平消融仍需区分新增CF监督与结构改动的收益。

### 1. 动机：不只是让真实关系比打乱关系高

旧正候选hinge只能要求正确候选自己的真实支持分高于对照分。若错误候选也获得同样增量，两者排名仍然不变；该损失甚至可为零。另一个实际风险是null选项垄断注意力：事实和对照都只读零关系，绑定差失去作用。v4因此同时处理“相对竞争监督不足”“空关系塌缩”“目标自身信息泄漏”和“无效反事实对照”。这些是待验证的工程与学习目标改进，不等价于已证明的因果识别或性能提升。

### 2. 数据流与各操作目的

设B为batch，M通常256个目标候选，P为原检测候选数，C=256，L通常64，D=128，R=min(64,P)。

```text
原编码器/检测解码器 → 原CRS → 原logit z[B,M,1]
                    ↓ 原目标特征Q[B,M,C]（可经过BCEM/Grab）
Q聚类8类 + CRS前3名所处簇 → 连续Top-N竞争范围（最少5、最多32）
                    ↓ q[B,K,D]，K为本batch最大N
原检测器Top64置信度 → IoU去重 → 参考v[B,R,D]（BCEM之前）
                    ↓ 每目标独立排除自身/重叠泄漏参考
参考语义 + 9维几何 + 目标 → 关系token[B,K,R,D]
                    ↓ 读取完整文本[B,L,D]
事实评分/注意力 → 对关注的至多4个合法参考交换中心位置，保留各自尺寸
                    ↓ 用同一评分器重新评分，过滤无效对照
Δ = 事实support - 有效对照support均值[B,K]
                    ↓ β=2 sigmoid(a)，初值0.1
z_final = z_CRS + βΔ → 只在连续Top-N中输出排序
```

1. **竞争范围不变。** 沿用v3确定性聚类、自适应连续前缀；离散选择不反传，不引入GT候选，不重新跑编码器/解码器。未进入前缀的候选仍计算完整定位BCE，但不输出、不参与该次反事实排序。
2. **参考池不变，防泄漏更严。** 仍按检测置信度先Top64再IoU≥0.8去重，不补满；参考特征来自原pre-BCEM解码器，经原refer_adapter。除同query和原IoU排除外，计算交集分别除以目标面积、参考面积，两者最大值≥`reference_coverage=0.5`时排除。目的：排除大框包住目标或小框只截取目标一部分的伪参考；并非要求所有参考IoU严格为0。
3. **稳定稀疏选择。** 合法参考+null一起做masked sparsemax。非法槽位用−∞参与排序，累计和时用0跳过无效尾部，避免旧−1e4占位值超过合法极低logit。始终保留合法null，避免全掩码softmax。
4. **参考专用软路径。** `a=(1-ε)*sparsemax(refs+null)+ε*softmax(legal refs)`，默认ε=0.1。有合法参考时，真实参考总权重至少ε；无合法参考时软路径退回null。目的：即使稀疏路径只选null，关系value仍有读取/学习路径。它保证的是总权重，不保证参考正确，也不保证每个注意力logit永不饱和。训练和测试使用同一规则。
5. **聚焦少量文本相关参考。** 按事实路径文本条件注意力，从合法参考中取至多`control_reference_topk=4`个，只在这些参考内部做确定性非原样循环置换，其他参考不动。选择截断梯度、不读GT。目的：避免大量低相关参考的改变稀释干预。早期注意力可能不准，仍须消融验证。
6. **位置干预与尺寸解耦。** 置换参考中心位置，但保持各参考原宽高。重新一致计算offset、方向、距离和目标尺度归一化offset；原9维中的宽高比保持。若任何合法参考交换后超出图像边界，整次对照忽略。目的：减少“交换整套几何后外观—尺寸不一致”的简单检测捷径，不伪造图像像素。
7. **检查对照有效性。** 除几何实际变化外，至少有一个被移动合法参考与交换来源的投影特征余弦距离大于`semantic_difference=0.05`。近乎相同的参考互换可能只是集合重排列，不能强迫网络制造差异。无有效对照时该目标Δ=0，训练CF也忽略；不因CF损失开关改变推理有效性规则。保留的对照仍不保证违反文本含义。
8. **融合不换评分范式。** support是未校准标量而非概率；Δ为事实支持相对有效位置对照的差，β可学习，最终仍保留原CRS logit。没有熵门控，没有把错误候选的Δ全部压成0；错误候选也可能具有真实关系，监督的是与正确候选的相对竞争。

### 3. 损失、标签与梯度

全候选BCE仍用原Hungarian refer标签：匹配query为1，其余为0，不把所有IoU高的框改为正例。

$$L = L_{\mathrm{det}}+L_{\mathrm{other}}+
1\cdot L_{\mathrm{CRS\ BCE}}+0.5\cdot L_{\mathrm{final\ BCE}}+0.1\cdot L_{\mathrm{CF}}.$$

原head的`loss_weight['refer']`仍乘前两项；CF只乘自身`cf_loss_weight`。检测、Grab/filter、BCEM、mask、correctness等原有权重不变。BCE每图按全部有效候选平均，再按全局有效图像平均，**不是只监督前N个**。

CF正例p：原Hungarian正query、进入Top-N、有有效对照、预测框与refer GT的IoU严格>`cf_positive_iou=0.5`。
CF负例n：同一图Top-N内原负query、有有效对照、与refer GT的IoU**恰好为0**，按原CRS logit选最多`cf_negative_topk=8`个。0<IoU≤0.5不参与CF负监督；没有合格正负对则跳过该图CF，不跳过BCE。

$$L_{\mathrm{CF}}^{(b)}=\frac{1}{|\mathcal P_b|}
\sum_{(p,n)\in\mathcal P_b}
\max\left(0,m+\operatorname{sg}(z_n^{CRS}-z_p^{CRS})
-\beta(\Delta_p-\Delta_n)\right),\quad m=0.2.$$

目的：让新增反事实证据足以克服原错误排序的分差，而非仅提高正确候选的绝对支持差。若原排序已满足margin且反事实不破坏它，CF可为0；不强迫所有样本额外增大β。这里不再额外叠加旧正例hinge。

- `detach_inputs=False`默认联合训练：最终BCE影响原CRS、候选/文本上游、关系评分器和β；CF影响关系评分器、β及共享候选/文本上游。
- CF式中仅原logit差截断，不能通过改变该项的原CRS分差来投机降低CF。CF不直接优化CRS评分网络参数，但共享上游会间接影响CRS。
- 框、聚类、参考筛选、有效性、正负选择均不反传；没有新增框回归。
- `detach_inputs=True`时新增分支截断候选、参考、文本与其基准logit，但原CRS BCE继续训练原模型。
- 多卡按全局CF有效图像归一化，某rank无正负对时仍参与相同collective并返回连通零梯度。

### 4. 配置、兼容与验证接口

当前完整配置：`configs/aerialvg/PropVG-aerialvg.py`，选择`birs_version="v4"`。切回`"v3"`选择指定实验的算法（CF=0、soft mix=0，CRS BCE=1、final BCE=0.5）。旧模型优先用归档配置测试。更早Git版本的0.5/0.5不是本次指定的v3训练权重；该权重修正不改变旧模型推理结果。

v4有独立schema=5及推理策略签名；参考coverage、特征差阈值、干预参考数、注意力/干预算法都进入签名。v3 schema=4不变。不能拿v3权重用v4配置直接评估/恢复；也不能更改v4这些推理参数后静默声称测试同一个模型。训练用margin/正负选择阈值/CF权重只影响训练，不进入推理签名。

接口延续`with model.head.birs.diagnostics():`与`extra_dict['birs_samples']`，保留全query索引、选中索引、合法参考、事实/对照support及注意力。新增/沿用`cf_valid_controls[B,K,P_cf]`表示真正计入Δ的对照；不可把所有`selected_controls`直接均值当作v4有效对照均值。

CF指标在v4中的含义：`cf_samples`为全局有合格正负对的图像数；`cf_gap`为这些图像的平均β(Δp−Δn)；`cf_margin_met`为满足原logit差+margin要求的对比例；`cf_positive_count`为参与成对监督的正query数；`cf_control_count`为这些正query的有效对照总数。常规BIRS字段与`loss_birs_final`继续按此前要求隐藏终端/txt展示，但不删损失和完整追踪记录。

测试脚本：`tools/test_birs_v4.py`。回归包含极低logit掩码、null塌缩读取、包含框过滤、尺寸保持、相同语义跳过、CF梯度边界、空GT/空参考/全padding、checkpoint重算、head训练/测试接线，以及直接读取指定实验归档源码对照v3默认前向和损失。

本次验证：170项CPU回归通过（包含v4专项15项及原CRV/BIRS/Grab/BCEM/correctness/两阶段关系/可选管线测试）；实际双进程Gloo的3个小型优化步中，一侧无GT、一侧无有效CF、全局无GT的参数梯度均与单进程合并batch一致。主配置可正常解析、构造v4并格式化保存。未运行真实数据集完整训练、CUDA训练或性能评估，不能据此声称性能提升。

### 5. 尚未解决的限制与实验要求

语义特征相近阈值只是代理，不能判定真实关系是否被破坏；文本注意力不等于参考物真值。参考特征本身编码位置，保留尺寸与边界检查只能减少部分捷径，无法消除特征内隐位置。严格不重叠负例也可能符合多解描述；CF系数仍需小量消融，不能承诺监督无噪声。

coverage可能误删真实重叠参考；Top-N之外的正确框无法被救回；无有效对照时回退CRS，不是强行制造反事实。所有超参数是可验证的初始设置，不是理论最优。禁止用test选择阈值或报告仅有利子集。

建议消融：相同1+0.5 BCE的无反事实控制；v4关闭CF；v4关闭参考软路径；关闭coverage/语义过滤；全部参考干预对比聚焦干预；同一checkpoint前后Top1改对/改错、Top5覆盖、null权重、CF有效样本数与β。文本打乱只作为敏感性诊断，不直接给打乱描述伪造定位标签。

---

## 历史方案：v3 自适应竞争范围与稀疏反事实关系细化

配置选择 `birs_version="v3"`；v1/v2的结构、参数布局和checkpoint策略仍保留，历史说明位于后文。v3是新结构，不能用v1/v2完整权重冒充恢复训练。关闭`enable_birs`仍沿用原PropVG/Grab/BCEM等流程。

v3默认对齐指定实验20261118_084936（CF=0、soft mix=0、BCE为1/0.5）。以下可选正候选CF与含null软混合是2026-11-18中间方案说明，不是v4算法或当前主配置；为可选配置保留接口。当前v4详见本文首节。

### A. 动机与边界

CRS已能将许多正确目标排进前几名，但外观相似的航拍实例仍可能竞争第一名。本模块在CRS之后确定一个自适应竞争范围，显式读取“周围有什么、在当前目标的什么位置”，再比较真实语义—位置绑定与打乱绑定的支持差异。它不重新预测框、不重新运行图像编码器/检测解码器、不解析文本、不引入外部模型或参考物GT。

“先看外观、再看关系”只是直观解释，CRS/检测特征本来已有上下文，BCEM开启时还有显式环境。两阶段和关系注意力本身不是原创；重绑定也只是特征对应对照，并非真实物理场景编辑或已识别的因果效应。该版是待验证方案，不保证性能提升。

### B. 数据流与每一步的目的

设B为batch、M为目标池大小（常用256，Grab开启可能减少）、P为原检测候选数、L为文本长度（常用64）、C=256、D=128。R=min(64,P)为参考槽位数，去重后用mask标记有效槽位；N_b是每图实际选择数，K=max_b(N_b)，全空时张量保留一个无效占位。

```text
BEiT-3 → 512通道共享SimFPN（不改）→ 原检测解码器
                                      ↓
                    检测queries [B,P,256] / 框 / 前景logits
                         ├─ refer_adapter → 原检测参考池（BCEM前）
                         │         ↓ 检测Top64 FIRST → IoU≥0.8去重
                         │      参考特征[B,R,256] / 框[B,R,4]
                         ↓
                  可选Grab → refer_adapter → 可选BCEM
                         ↓
               目标特征Q[B,M,256] → 原CRS(Q,T)
                         ↓ 原logits/probability [B,M,1]
           Q聚类 → 多个高分种子簇 → 有限搜索 → 连续Top-N_b
                         ↓
             选中目标[B,K,256] / 框[B,K,4] / valid[B,K]
                         ↓
        目标投影 / 参考投影[B,R,128] / 9维相对几何[B,K,R,9]
                         ↓
          关系对编码[B,K,R,128] → 读取完整文本[B,L,128]
                         ↓
         sparsemax选择R条关系 + 1个零值空选项
             可选混合少量masked softmax权重
                         ↓
                真实支持r_f[B,K]
                         ↑ 同一评分器、重新计算注意力
         合法参考物内部置换几何对应 → 对照支持r_cf[B,K,P_cf]
                         ↓
              Δ=r_f-mean(r_cf) → βΔ（不做组内中心化）
                         ↓
       回填全M个logit（集合外δ=0，专供完整定位监督）
                         ↓
      仅选中Top-N_b按最终logit排序并输出；不附加集合外候选

训练额外：原Hungarian匹配的正query在Top-N内且有有效置换
          → 真实支持 - 有效对照平均支持
          → 正间隔hinge loss（不乘β，不增加GT候选）
```

1. **自适应竞争范围，目的：不预先假设每图只有固定数量竞争者。** 对有效Q做确定性球面K-means（默认8簇、6轮），聚类只产生离散标签，不计算共享原型。按原CRS概率降序，取前`seed_topk=3`涉及的簇；仅在前`max_targets=32`搜索这些簇的成员。最远排名确定N，至少保留`min_targets=5`，不足时保留所有有效候选。保留连续前缀，因此排名1/3/5/6/7同簇时第2/4名也保留。新增中间成员不递归扩展其簇。种子包含多个簇，避免Top1孤立误判排除另一竞争簇。选择不读GT、不反传梯度。BCEM开启时Q是增强后的候选表示，不能声称是纯外观聚类。
2. **独立参考池，目的：参考物不必像目标。** 原检测分类softmax的前景通道0取Top64，再贪心抑制IoU≥0.8，不从64名外补齐。v3无论Grab是否开启，都使用原解码器的refer_adapter特征，位于BCEM之前；目标侧保留原Grab/BCEM流程。同源query及目标IoU≥0.8参考物排除。框、置信度筛选及来源索引截断梯度。
3. **相对几何，目的：明确物体的空间归属。** 使用既有9维编码：短边归一化dx/dy、单位方向2维、log(1+距离)、log宽高比2维、目标尺寸归一化位移的tanh共2维。框是有效图像归一化cxcywh，转裁剪xyxy后使用`img_shape`，不是padding画布；这是二维图像几何，不是米制三维位置。
4. **候选自身条件，目的：每次明确正在评估谁。** 目标q_i、参考v_j分别线性投影为D维，几何经MLP投影为D维。拼接`[q_i,v_j,g_ij,q_i*v_j]`，经Linear/GELU/LayerNorm编码为关系token。不使用Top-N均值原型、簇均值或不同目标之间的自注意力。
5. **完整文本交互，目的：关系证据按描述读取。** 关系token为Query，投影文本为Key/Value，一层多头交叉注意力（4头，dropout=0）加残差LayerNorm；padding不参与。输出[B,K,R,D]。没有显式“黑车”类别标签或词—框监督，不能保证注意力就是正确参考物识别。
6. **稀疏汇总，目的：避免64个参考平均稀释一两个关键证据。** 每条交互后关系token预测选择logit，额外空选项logit由目标+有效文本平均表示预测。先对合法关系与空选项一起做masked sparsemax；空选项value固定为0。可选 `attention_soft_mix=ε` 将权重改为 `(1-ε)*sparsemax + ε*masked_softmax`，两者使用完全相同的合法掩码。v3当前配置和旧配置缺省ε=0，沿用原计算，不额外乘加；ε=0.1属于中间版可选实验，与v4参考专用soft不同。加权关系表示与目标、句子拼接，经MLP输出一个support标量。

   **软分量的目的：** 纯sparsemax若只选择空项，真实和打乱路径都读取零关系，可能形成精确零梯度死区；即使增加CF损失，也可能无法恢复。小比例softmax保留合法参考的可微读取路径，不删除空项、不把非法参考补回来，不强迫选固定数量。softmax仍可能数值饱和，因此这不是梯度永不消失的保证。该混合在训练和推理时相同，不能只在训练中加软分量而测试时偷偷换回纯稀疏。
7. **重绑定对照，目的：控制“有什么”和“有哪些位置”，只破坏对应。** 对每个目标独立在其合法参考集合内部循环置换几何；语义、目标、文本及几何集合不变，不把无效/自身槽位引入。相同评分器重新执行关系编码、文本交互、稀疏选择和support评分，不搬用真实注意力。默认2次对照，偏移由原目标query ID与对照编号确定，训练/测试一致、不使用GT/RNG，合法数≥2时非原样，少量参考时对照可能重复。它不是均匀随机置换的精确期望，也不保证是最强破坏，属于可消融的确定性采样策略。
8. **差值与融合，目的：提取真实绑定相对对照的额外支持，保留第一阶段匹配。**

$$\Delta_i=r_i^{f}-\frac{1}{P_{cf}}\sum_p r_i^{cf,p},\qquad
\ell_i^{final}=\ell_i^{CRS}+\beta\Delta_i,\qquad
\beta=\beta_{max}\sigma(a).$$

β初始0.1，上限2；不再做组内中心化，不增加门控。少于2个合法参考、全padding文本或无效目标时δ=0。刚启用尚无有效训练步的eval也令δ=0，但仍执行自适应前缀选择；因此v3未训练评估不是完整256框的原基线。推理不强制真实分高于置换分；可选CF损失仅训练时约束合格正候选的平均支持差。

### C. 损失与梯度：完整定位监督与可选CF间隔

#### C.1 原CRS与最终定位BCE

设v为全部M个有效候选掩码、y为原Hungarian refer标签。单个refer GT对应的匹配query标1，其余query标0，**不是所有IoU>0.5的框都标1**。两阶段都计算全部有效候选上的BCE，每图除以有效候选数，再按全局有效样本数平均。BCE有效样本只要求原refer GT存在且匹配正候选有效，不要求正候选进入Top-N或有参考物。

v3按指定实验保留：`loss_refer=BCE_CRS*w_refer`，`loss_birs_final=final_mix*BCE_final*w_refer`，默认1/0.5。原始refer BCE在REFHead中被这一项替换，不再额外重复加一次原始项。v4保持相同BCE系数，见首节。

集合外logit保持原CRS值，也继续接受完整定位BCE监督，不缩小为Top-N-only BCE。`loss_det`及原Grab/filter、mask、correctness权重不改。

#### C.2 只监督匹配正候选的反事实间隔

**目的：** 最终BCE允许CRS自己完成任务、关系残差保持零；新增损失直接要求正目标的真实关系比置换关系更有支持力，不依赖小β传回弱梯度。

对选中的正候选i，仅保留合法参考数≥2、图文有效，且确实改变几何数值的置换集合V_i。代码检查合法槽位上几何变化的最大绝对值是否大于1e-6，检查不求导。只有索引变化但几何不变的对照不参加CF平均。选中集合由原预测确定，**不向其中塞入GT框或强行补进正query**。

$$
g_i=r_i^f-\frac{1}{|V_i|}\sum_{p\in V_i}r_i^{cf,p},\qquad
L_{CF,b}=\frac{1}{|P_b|}\sum_{i\in P_b}\max(0,m-g_i),\quad m=0.2.
$$

P_b为Top-N内满足上述条件的原匹配正query，通常至多一个。这里使用**未乘β的原始支持差**。达到间隔后hinge为零，不继续把支持差推向无穷。正query不在Top-N、没有GT、参考不足、全padding、没有有效置换时，该图跳过CF项，但完整BCE仍按自己的有效性规则计算。不为负候选增加反向间隔，不把所有真实关系标1、所有打乱关系标0。

推理残差仍使用原来的全部配置置换均值；CF项仅对有效对照求均值。因此在部分置换无效的退化情况中，CF辅助差值与推理差值可能不同，最终BCE仍负责融合校准。

#### C.3 总损失、梯度与多卡归一化

$$
L=L_{det}+w_{refer}(L_{CRS}+0.5L_{final})+\lambda_{CF}L_{CF}+L_{其他原有},\qquad\lambda_{CF}=0.1.
$$

`loss_birs_cf`已乘 `cf_loss_weight`，不再乘β或refer权重；它与原 `loss_birs_binding` 不是同一个损失。`cf_loss_weight=0`完全关闭该项，缺省就是0，旧配置不会增加新训练目标。最终BCE还会继续优化β；CF本身没有通向β或原CRS logit的非零直接梯度，而是更新共享的关系评分器。`detach_inputs=False`时CF还可影响目标/参考候选、文本的上游，包括目标侧BCEM；它不直接经过CRS评分网络，但共享输入的变化仍会间接改变CRS。

选择/框/置换不求导。`detach_inputs=False`保持联合优化；True时CF及最终分支截断其输入，第一阶段BCE仍正常训练原模型。沿用现有梯度裁剪逻辑。

多卡按BCE、CF各自的**全局有效图像数**归一化，不让没有CF正候选的rank稀释损失。CF开启时计数及其gap/达标率统计合并为一次all_reduce；关闭时保留原标量collective。无监督rank保留参数连接零梯度。CF有效样本是BCE有效样本子集，原 `trained_steps` 更新规则不变。

**监督边界：** 几何发生变化不代表表达式一定变假。无关系描述、同义参考或多解场景下，正间隔仍可能引入噪声；本版未解析文本，也没有额外关系GT来证明置换是有效负例。低权重和hinge只限制推动幅度，不能消除这一假设，需要对照验证。

### D. 只输出Top-N，不处理集合外的排序

训练内部保留[B,M,1]以计算全候选BCE；前向返回`output_mask/candidate_output_mask`。`MIXRefUniModel_OMG.get_predictions_parts`在原退化框过滤后，用同一掩码过滤框、检测分、referring分和排序值，然后才执行可选NMS/rescale。输出是每图实际N个有效框，不把原后缀接回去，训练/验证/测试后处理一致。Top-5在N≥5时正常评估，不足5个有效框时按现有评估取可用候选。既有非v3不提供此掩码，沿用原后处理。

### E. 配置与兼容

```python
enable_birs = True
birs_version = "v3"
birs = dict(version="v3", num_groups=8, kmeans_iters=6,
            seed_topk=3, min_targets=5, max_targets=32,
            reference_topk=64, duplicate_iou=0.8,
            model_dim=128, num_heads=4, chunk_size=8,
            permutations=2, initial_strength=0.1, max_strength=2.0,
            final_mix=0.5, detach_inputs=False, checkpoint_chunks=True,
            cf_loss_weight=0.1, cf_margin=0.2, attention_soft_mix=0.1)
```

`final_mix`控制新增融合BCE权重；`cf_loss_weight`控制CF是否开启及其系数；`cf_margin`是正候选支持间隔；`attention_soft_mix`控制软注意力比例，不依赖CF损失开关。若要完全恢复修改前v3行为，两项均设0：`cf_loss_weight=0, attention_soft_mix=0`。`permutations`是置换次数；`checkpoint_chunks`只以重算换训练显存。分块只计算至本batch最大实际N，不固定计算至32；仍有batch padding开销。主配置没有改动用户的teacher、Grab、BCEM、工作目录及其他开关。建议新实验使用独立work_dir。v1/v2分支不接收v3参数，主配置已按版本构造不同字典，不能仅将旧归档字典的version改成v3。

v3参数名称、形状和schema=4保持不变，不增加可训练参数或持久化buffer。`attention_soft_mix`缺省/等于0时，原策略摘要逐字节不变；非零值会加入摘要，因为它改变推理。`cf_loss_weight/cf_margin`只影响训练损失，不加入推理策略摘要。旧模型按原归档配置或soft mix=0测试，仍可通过原校验；用soft mix=0.1的新配置测试纯稀疏旧模型会明确报错，而非静默改变预测。反过来，新混合注意力模型也必须保持训练时mix测试。

新实验可从相同BEiT-3初始化训练；完整baseline初始化用`--finetune-from`。旧BIRS模型若要改成非零soft mix继续训练，属于新注意力策略，当前加载器不会绕过摘要保护；需另行明确设计权重迁移，不是无缝resume。v1 schema=1、v2 schema=3及关闭BIRS的旧模型路径保持。v3目前接入当前AerialVG使用的MIXRefUniModel_OMG，不宣称其他未接入head具有同样输出行为。

使用当前代码测试旧v3时优先用它自己的归档配置，或在配置源码将`birs_version="v3"`后重新解析配置。当前主配置是v4，不能仅用`--cfg-options`把soft mix/CF改成0就当作v3，也不能只改`model.head.birs.version`而留下v4专用字段。完整结构参数、版本和推理策略均须与旧实验一致。

启动方式不变：`python tools/train.py configs/aerialvg/PropVG-aerialvg.py`，或现有多卡脚本。未自动启动完整训练。

### F. 可验证接口、日志与实验要求

使用原`with model.head.birs.diagnostics():`，在`extra_dict['birs_samples']`读取CPU诊断：全query索引下的groups、candidate_mask、delta、raw_binding（此处为support差值）、paired/null（真实/平均对照support）、原/最终logits、参考池及合法掩码；`target_indices/target_valid`映射变长目标；`selected_attention[B,K,R+1]`、`selected_control_attention[B,K,P_cf,R+1]`、`selected_support[B,K]`与`selected_controls[B,K,P_cf]`用于比较两条路径。支持分是未校准标量，不是概率。未开启diagnostics不保存所有对照注意力到CPU。

主要日志：`birs_target_count`平均N、`birs_target_cap_rate`达到配置/候选数上限比例、`birs_refer_in_topn`原匹配正候选覆盖、`birs_reachable`IoU≥0.5覆盖、`birs_reference_count`去重参考数量、`birs_null_attention`平均空选项权重、`birs_sparse_reference_count`非零参考数量、`birs_delta_abs`按选中目标平均绝对修正、`birs_strength`、`birs_eligible`选中目标可用绑定比例、`birs_corrected/harmed/rank_changed`训练batch的Top1变化、`birs_supervised_samples`全局有效样本数。日志不能代替完整val诊断。

2026-11-18按需精简展示：上述常规BIRS统计和 `loss_birs_final` 不再输出到终端/txt的逐batch训练日志，完整值仍保留在云端训练指标及模型诊断接口中。新增CF损失和CF相关字段仍显示。这里只过滤日志字符串，不删除损失计算、梯度、全局归一化或trained_steps更新。

开启软混合时，`birs_sparse_reference_count`仍统计原sparsemax分量选中的数量，避免所有softmax非零项把“稀疏选择数”虚增；另增 `birs_mixed_reference_count`统计混合后实际非零参考数。CF开启时新增：

| 日志 | 含义 |
|---|---|
| `loss_birs_cf` | 加权后的CF hinge，已经乘0.1 |
| `birs_cf_samples` | 当前全局batch中有合格正候选与有效置换的图像数 |
| `birs_cf_gap` | 这些图像正候选的平均真实减有效对照支持差，未乘β |
| `birs_cf_margin_met` | 支持差达到margin的正候选比例（先图内平均，再全局图像平均） |
| `birs_cf_positive_count` | 每个CF有效图像的合格正query数，通常1 |
| `birs_cf_control_count` | 每个CF有效图像的正query有效对照总数，通常2 |

gap、达标率与计数按全局有效样本归一化，无CF样本时显示0。CF有效数小于全局batch不一定报错，应先看正query是否进入Top-N、合法参考数及几何变化。

必须比较：固定N/自适应N；真实关系直接评分/真实减置换；普通注意力/稀疏选择；BCEM开关。比较最终分与同checkpoint原CRS时，应同时报告完整CRS与同一自适应前缀口径，不能将筛选召回损失误认为重排收益。跨独立模型仍需同训练配置、同损失总权重及多种子验证。重点统计CRS错但集合可达的样本中残差朝GT推动比例，而非仅观察总体loss。

潜在局限：种子覆盖多簇可能使N长期取满上限；聚类尾部成员可能是噪声；参考漏检/误检；稀疏选择早期全部偏向空项；网络忽略绑定或关注伪线索；确定性少量置换不充分；无关系/多解描述难以产生有用差值；联合梯度可能干扰原模型；β小导致差值网络学习慢。不能为保证下降而把所有置换强行标成负例。

中间版CF与软分量不保证能纠正错误参考、消除描述歧义或超过基线。指定v3默认BCE为1/0.5；跨版本消融应匹配损失总权重，避免将重加权收益混入机制结论。

### G. 实现验证记录

2026-11-18新增CF验证：

- 共154项CPU单元/回归测试通过：`PYTHONPATH=.:tools python -m unittest test_crv test_birs test_birs_v3 test_optional_pipeline test_grab_paper test_bcem_ray test_correctness test_two_stage_relation`；覆盖旧BIRS v1/v2、CRV、BCEM、Grab、correctness及原模型分支。
- 新CF的hinge方向、达标后停止梯度、只监督匹配正query、不乘β、无正候选/无GT/无参考/全padding/无变化对照跳过，以及联合/截断梯度均有单元覆盖；纯sparsemax全空项时梯度为零、加入软分量后几何评分路径获得非零梯度也已验证。
- 已实际加载11/16单独BIRS3实验的归档类，向当前默认设置严格加载其状态并校验摘要；在相同输入与权重下，8项推理张量（含分数、残差、排序、支持与注意力）与归档实现逐值完全一致。此处没有跑完整数据集测试。
- 新CF的两进程Gloo三步验证通过：一侧无GT、另一阶段一侧有GT但无CF参考、最后全局无监督；每步梯度均与单进程合并全局batch参考一致。
- CUDA B=2、256候选、64文本token、C=256/D=128，开启分块checkpoint与CF，前向和全部新增参数/输入反向梯度有限；有效CF图像数为2。只做模块验证，没有启动正式训练或改动旧模型权重。

以下为初版v3历史验证记录，当前损失系数及可选CF以C/E节为准：

- 119项CPU单元/回归测试通过：`PYTHONPATH=.:tools python -m unittest test_birs_v3 test_birs test_optional_pipeline test_grab_paper test_bcem_ray test_correctness test_two_stage_relation`。含多种子/连续前缀/禁止递归扩簇、变长batch、置换合法集合、稀疏空项、全候选BCE/集合外正例、联合与截断梯度、无GT/全padding/无参考/全退化框、checkpoint与分块、真实REFHead训练和推理、输出裁剪/NMS/rescale/correctness对齐及既有模块回归。
- CUDA小规模模块测试：B=2、256候选、64文本token、C=256/D=128，checkpoint开启，前向、两阶段BCE、反向梯度均为有限数；随机输入选中N=30/29。只证明实现可运行，不代表实际数据N分布或完整模型耗时。
- 两进程CPU Gloo DDP两步通过：一个rank没有GT，固定collective正常，全局监督样本数为1，所有新参数梯度存在且有限，优化后两rank权重逐值一致。
- 上述不是完整数据集训练或性能验证，没有自动启动正式训练，也没有给旧实验模型替换权重。

## 历史版本说明：v1 / v2

## 1. 当前方案与动机

当前通过 `birs_version` 区分 **v1：第一版外观聚类** 与 **v2：第二版Top-N关系细化 + 全候选融合BCE**。默认v2，保持修改前的流程与权重策略。BIRS 为 Binding Interaction Residual Scoring（关系绑定交互残差评分），放在 CRS 之后，不替换 CRS，不修改预测框，不重跑图像编码器或检测解码器。

### 两个版本的精确定义

| 项目 | v1：第一版 | v2：第二版（当前默认） |
|---|---|---|
| 绑定计算对象 | 全部有效候选，通常256个 | 原CRS Top-N，默认5个 |
| 分组 | 确定性球面K-means，默认8簇、6次更新 | Top-N为一个竞争集合，不执行聚类 |
| 原型 | 各簇有效候选特征均值 | Top-N特征均值 |
| 中心化 | 各簇内有效绑定候选分别中心化 | Top-N内有效绑定候选中心化 |
| 最终排序 | 全部有效候选按融合logit全局排序 | 仅重排Top-N内部，外部原顺序不变 |
| 融合BCE候选范围与分母 | 全部有效候选 | 全部有效候选，**不是Top-N-only BCE版** |
| 融合BCE样本条件 | 有有效匹配正候选，且至少一个候选可产生绑定 | 有有效匹配正候选，不要求存在绑定 |
| 绑定排序监督 | 原匹配正候选与同簇IoU=0的hard negatives | Top-N内原匹配正候选与IoU=0的hard negatives |
| 对同checkpoint的Top-5影响 | 允许改变Top-5成员 | N=5时成员不变 |

v1从 `20261112_235348` 归档实现恢复，包括原确定性聚类、全局排序及无绑定时跳过新增BCE的规则，不是把所有候选平均成一组，也不是只保留平均分最高的一簇。两版共享参数结构、参考物选择及绑定公式。BCEM、Grab、teacher等开关独立，切换BIRS版本不会改变它们；不开Grab时，两版参考特征均沿用当前接入方式（BCEM开启则参考特征也经过BCEM），本次没有额外改变参考支路。

下面第3–5节的数据流与Top-N细节主要描述v2；v1对应差异以上表为准。

航拍小目标的细粒度外观信息有限，密集场景中经常存在颜色、类别、尺度相似的实例。第一阶段匹配可以找到“像描述的一组目标”，但不一定确定具体实例。广域俯视/斜俯视图像又允许观察较多物体及其配置，因此参考物的语义与相对位置可能提供补充证据。

核心问题不是分别判断“有黑车”和“左侧有物体”，而是判断“黑车是否绑定在左侧”。实际绑定相对于重绑定对照的增益，用于补充原图文匹配。

**v2为何不用外观聚类？** 外观特征相似不等于竞争当前描述。上一轮 epoch34 诊断中，test 有832条“CRS错误但Top-5含正确候选”的样本，正确候选与错误Top-1同簇仅271条。v2直接用 CRS Top-N 定义文本条件下的竞争集合，避免聚类边界限制绑定排序监督；这不是已经证明v2优于v1的结论，两版保留供实验对照。

Top-N 筛选和中心化本身不是创新；关系绑定与条件竞争集合的组合需要消融和查新验证，不能保证原创或性能提升。这是针对航拍困难的设计，不是只适用于无人机的数学机制；重绑定只是统计关联对照，不是已识别的真实因果效应。

## 2. 与现有模块的关系

- 原 CRS 始终保留，原定位损失照常监督。
- 可选 Grab / BCEM 的原流程保持；BCEM 在 CRS 前增强表示，BIRS 不读取其空间路径或图像特征图。
- Grab 若筛掉部分目标候选，参考支路仍读原检测池的 refer_adapter 特征，不从 Grab 筛选结果裁剪参考。
- 不需要 teacher、关系标签、额外参考物 GT、文本人工拆解或外部模型。
- 旧 CRV / two-stage relation refinement 已移除，不再叠加这些重排器。
- 关闭 enable_birs 时不构建新模块，不增加损失、参数或collective。

## 3. v2完整数据流与维度

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

## 4. v2每一步的目的与精确规则

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
birs_version = "v2"        # "v1" 第一版聚类；"v2" 第二版Top-N
birs_target_topn = 5
birs_num_groups = 8
birs = dict(
    version=birs_version,
    target_topn=birs_target_topn,
    num_groups=birs_num_groups, kmeans_iters=6,
    reference_topk=64, duplicate_iou=0.8,
    model_dim=128, num_heads=4, chunk_size=32,
    initial_strength=0.1, max_strength=2.0,
    final_loss_weight=1.0, binding_loss_weight=0.1,
    negative_topk=8, detach_inputs=False,
)
```

`target_topn`只影响v2；`num_groups/kmeans_iters`只影响v1。要复跑第一版，设置`birs_version="v1"`，其余配置保持对照一致。默认版本为v2，旧的v2配置不写version也仍使用v2；第一版旧归档配置测试时必须显式补上`model.head.birs.version="v1"`。旧CRV和relation_refinement已移除。启动方式不变，建议设置独立work_dir：

```bash
conda activate propvg
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/aerialvg/PropVG-aerialvg.py
```

兼容范围：

- enable_birs=False（旧配置没有此参数时也默认False）：仍沿用原PropVG/Grab/BCEM等已有路径，既有非BIRS模型可以继续训练/测试，不要求新增BIRS参数。correctness观察头现已移除。
- v1恢复第一版聚类计算，使用原schema=1及原策略摘要，可加载匹配的第一版权重；v2保留schema=3及其全候选损失/target_topn摘要，现有修复版v2权重可继续加载。两版显式校验策略，不能交叉加载或悄悄切换排序语义；中间的Top-N-only-BCE/schema=2仍不支持。
- 配置中的用户版本v1/v2与checkpoint内部schema=1/3不是同一套编号。不再提供candidate_mode别名。
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
4. v1/v2各自的checkpoint策略校验；验证False路径的旧参数/RNG/前向/梯度不受影响，既有非BIRS权重可严格加载。
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
- v2只修正全候选融合BCE口径，没有增加门控、损失类型、外部监督或文本解析。v2推理仍有硬边界，BCE直接监督logit而非排序操作本身。v1/v2切换用于明确比较聚类与Top-N方案，不能保证任一版本性能更高。

## 9. 本次验证范围

106项测试通过，覆盖两个版本的分支选择、checkpoint隔离、v1聚类/空输入/梯度、v2全候选BCE及其梯度、正候选在Top-N外、没有绑定参考时的监督保留、Top-N硬边界、独立参考池及原PropVG/Grab/BCEM/correctness等回归路径。

另直接读取第一版实验归档源码，与恢复的v1比较：residual、paired、crs_control三种绑定模式，分别在联合/截断梯度两种设置下，初始化state_dict（含策略摘要）、前向张量、损失及原有日志指标均逐值一致。此处为小输入CPU验证，不是重新训练或完整数据集性能验证。

GPU两样本测试验证联合/截断梯度下全256候选BCE；集合外正logit的梯度与全候选平均公式一致。两进程CPU DDP连续两步通过：两个rank都参与融合BCE，只有一个rank存在绑定配对，参数梯度仍同步。加入BIRS前BASE归档配置及真实检测头权重严格加载，初始化RNG和检测头推理输出与归档实现逐位一致。

本次未启动完整训练，未停止用户运行中的实验，也未修改既有实验归档。
