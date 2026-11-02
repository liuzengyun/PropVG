# 候选中心成对关系评分与自适应证据聚合

**模块名称：** 候选中心关系评分头（Candidate-Centred Relation Scorer）。

**当前方案：** 每个候选独立作为目标假设，从去重后的检测上下文中选择最近的K个邻居，默认K=6。将当前候选、单个邻居、有方向的5维几何和完整文本送入共享配对网络，得到K个匹配logit及K个采用权重，最后加权形成该候选的定位logit。开启时替换CRS，不与CRS分数融合；关闭时仍使用原CRS。

代码：`propvg/models/heads/candidate_relation_scoring.py`。接入：`REFHead.py` 的 `_score_referring`。版本：`schema_version=1`。

## 1 研究动机与问题定义

### 1.1 从候选自身匹配到目标假设周围的关系证据

航拍场景常包含多辆外观相近的车。描述“黑色SUV左侧的白色轿车”既限定当前目标的外观，也限定其与另一物体的配置。只看当前候选的类别、颜色未必能区分多个白色轿车。

原PropVG图文骨干和解码器已经包含隐式上下文交互，本模块不假设其没有关系能力，而是在评分阶段显式绑定“某个邻居的视觉内容”和“该邻居相对目标假设的位置”，检验这种绑定是否能提供额外区分信息。

### 1.2 固定目标角色，逐对判断而非局部集合交互

对每个候选j，依次以一个邻居i构成有方向的配对。评分问题是“以j为目标、i为上下文时，是否支持完整描述”，不是“j和i是否有关联”。共享评分器分别处理K组输入，邻居之间没有自注意力、图消息传递或多跳推理。

因此本版不是“目标token加6个邻居token共同交互”，也不运行旧CRS来预选高分候选。每个候选对应K个成对logit，自适应聚合后再对所有候选排名。

### 1.3 与背景增强的分工

BCEM从框外空间特征读取连续环境观测，并回写增强候选特征；关系头组织离散预测物体及几何，输出分数，不回写候选特征库。两者可以组合，也可以分别开关。没有开启BCEM时，关系头直接使用原候选表示。

这是一种可验证的评分结构改进，尚不能仅凭实现或注意力可视化声称具有充分创新性或性能优势。

## 2 符号、输入和总体dataflow

记batch为B，实际候选数N，原通道C=256，文本长度L通常64，邻居数K=6，评分维度D=256，几何维度G=64。

输入是经过原`refer_adapter`及可选BCEM后的候选特征、同索引预测框、同索引检测分类logit、完整文本与padding mask、图像有效尺寸。若Grab先筛选，N是实际保留数而非固定256；AerialSense也支持128个初始候选。

```text
图像 + 完整描述
 → 原BEiT-3、SimFPN、Deformable DETR
 → 可选Grab → refer_adapter → 可选BCEM
 → 当前候选特征库 V：B×N×256
    │
    ├─关系评分关闭：原CRS → 原分数、原损失、原排序
    │
    └─关系评分开启：不构建/执行旧CRS
       ├─预测框与检测前景分数detach → 上下文去重
       ├─对每个目标假设j：排除自身/近重复 → 选最近K个
       ├─当前候选 + 一个邻居 + 5维相对几何 → 配对token
       ├─每对独立读取完整文本，两层交叉注意力+FFN
       ├─共享得分头 → K个匹配logit
       ├─共享权重头 → K个采用logit → 有效邻居softmax
       └─加权聚合 → 每个候选一个定位logit
          → 训练：沿用原refer标签，BCEWithLogits
          → 推理：全部候选按新logit排序，最高分框作为结果
```

代码不接受GT作为关系头输入。teacher、辅助GT和关系标签都不是该头的前向依赖；现有teacher开关仍只按原流程影响检测监督，不因新增模块改变。

## 3 上下文去重与6近邻选择

### 3.1 只对上下文库去重，不删除目标候选

**目的：** 避免同一个物体的多个预测框占满邻居名额。

从当前DETR输出的两类分类logit取`softmax[...,0]`作为前景置信度。框按有效图像裁剪，排除非有限值及裁剪后零面积框。按前景分数从高到低贪心保留上下文代表；与已保留代表IoU≥`duplicate_iou`的框被上下文抑制，默认阈值0.8，置信度并列保持原索引顺序。

该筛选只产生上下文索引库，不修改检测框、检测标签或参与最终评分的N个候选。即使j在上下文去重时被另一个框代表，j本身仍会获得独立定位分数。

### 3.2 对每个目标假设选择最近邻居

**目的：** 保留邻近物体配置，同时控制每候选计算量。

对j排除：自身、与j的IoU≥0.8的上下文代表。再按有效图像像素坐标中的中心欧氏距离，取最近的K个；距离并列保持原索引顺序。本版不加最大半径、不加前景分数硬阈值、不按CRS或新评分预筛。

前景置信度只用于重复框代表的选择，不保证保留下来的每个上下文都是物体。因此仍可能选到背景预测，也可能遗漏距离较远的真实参考物。

### 3.3 有效性与空邻域

**目的：** 不重复填充邻居、不制造不存在的关系、不产生全负无穷softmax。

不足K个时，索引用-1、几何填0，以`neighbor_valid`屏蔽补齐项，其权重为0。完全无有效邻居时，只启用第一个“零上下文”配对：邻居特征及几何输入为0，其余K−1项无效。它复用同一配对评分器，退化为当前候选与文本的判断，不新建未训练的独立回退头。

`relation_effective_mask`在空邻域首位为True，`relation_neighbor_valid`仍全部False，`relation_fallback=True`明确标记此情况。无效目标框也使用有限回退评分，最终仍由原后处理删除退化框。

## 4 有方向的5维几何

### 4.1 坐标系、方向与尺度

**目的：** 防止padding错位，并保留图像长宽比下真实空间方向。

输入为相对有效图像归一化cxcywh。先转xyxy并裁剪到[0,1]，乘以有效图像W、H得到像素框，再由裁剪后的框计算中心与宽高。`pad_shape`不参与几何。

取统一尺度：

$$D_I=\sqrt{W^2+H^2},\quad
\Delta x_{ji}=\frac{c_{x,i}-c_{x,j}}{D_I},\quad
\Delta y_{ji}=\frac{c_{y,i}-c_{y,j}}{D_I},\quad
d_{ji}=\sqrt{\Delta x_{ji}^2+\Delta y_{ji}^2}.$$

横纵方向共用同一尺度，不分别除以W和H，以免改变距离的长宽比。图像横轴向右、纵轴向下，因此正dx表示邻居在右侧，正dy表示邻居在下方。

### 4.2 形状比例与几何编码

**目的：** 将物体间大小关系与相对位置共同提供给配对评分器。

记归一化宽高为w/D_I、h/D_I：

$$r_{ji}=\left[\Delta x_{ji},\Delta y_{ji},d_{ji},
\log\frac{w_i/D_I+\epsilon}{w_j/D_I+\epsilon},
\log\frac{h_i/D_I+\epsilon}{h_j/D_I+\epsilon}\right].$$

默认epsilon=1e-6，两个log比值裁剪到[-5,5]作数值保护，不是语义筛选阈值。逆转目标与邻居时，位移和log比值变号，距离不变。

几何经Linear(5,64)→GELU→Linear(64,64)→LayerNorm得到g_ji。几何可学习投影正常训练，但坐标及离散索引停止梯度。

## 5 成对图文评分网络

### 5.1 非对称配对表示

**目的：** 区分被评分的目标假设和作为证据的邻居。

候选共享LayerNorm(C)→Linear(C,D)投影，文本用独立同维投影。组合：

$$u_{ji}=P\left([v_j;v_i;v_i-v_j;\phi(r_{ji})]\right),$$

其中P为Linear(3D+G,D)→GELU→LayerNorm。默认输入832维、输出256维。固定拼接顺序与有符号差值明确目标/邻居角色，不做无方向平均。

### 5.2 两层完整文本读取

**目的：** 让每条物体关系直接与完整描述匹配，不解析主语、参考物或关系短语。

每个配对只有一个查询token。每层以该token为Query、完整文本为Key/Value，经过8头交叉注意力、输出投影、残差LayerNorm、512维FFN与残差LayerNorm。默认两层，dropout=0.1；没有邻居集合自注意力，也不复用CRS参数。

文本Key加入一维正余弦位置编码，Value保留投影文本内容；padding不参与读取，整数mask先转bool。全padding或零长度文本使用有限的空读取，不从填充值获得证据。

## 6 自适应聚合与最终定位

**目的：** 允许单一强证据或多个邻居共同参与，而不是硬max只更新一个分支。

共享配对表示h_ji输出两路：

$$s_{ji}=f_s(h_{ji}),\quad a_{ji}=f_a(h_{ji}),\quad
\alpha_{ji}=\operatorname{softmax}_{i\in\mathcal N(j)}a_{ji},$$
$$S_j=\sum_i\alpha_{ji}s_{ji},\quad p_j=\sigma(S_j),\quad
j^*=\arg\max_jS_j.$$

得分MLP为256→128→1，权重MLP为256→64→1，均使用GELU。权重头最后一层weight和bias初始化为0，使初始有效邻居权重均匀；不强制稀疏，不加熵正则或新监督。

聚合的是logit，不是sigmoid概率。上下文索引在特征gather时保持一一对应；交换邻居顺序应不改变最终分数。推理通过已有`ranking_priority`传递新logit，避免sigmoid饱和后并列导致排序精度丢失；关系头关闭时，原CRS分支的历史排序完全不改。

权重是训练得到的采用系数，不是“邻居确为参考物”的监督概率，也不能仅凭高权重认定学到了真实关系。

## 7 损失与梯度边界

**目的：** 单独检验评分结构，不同时改变标签、检测损失或数据监督。

使用原DETR最后一层匹配结果产生的refer正负标签；开启Grab时沿原索引收集标签。不顺带改成“所有IoU≥0.5的框均为正”，不对每个邻居额外赋标签。

$$L_{refer}=\lambda_{refer}\operatorname{BCEWithLogits}(S,y),\qquad
L=L_{det}+L_{refer}+\text{原开关启用的其他损失}.$$

新头复用原`loss_refer`字段，不增加独立loss。原检测分类/L1/GIoU、伪框读取、filter和mask逻辑不改。`allbbox`、`refer`等权重沿用运行配置，不因开启模块自动改变；不同配置的检测权重可能不同，对照实验应主动检查。损失中的`L_det`指已经乘过原权重的检测损失。

特征和文本正常回传，定位BCE可优化新头、BCEM及前级共享网络。坐标、前景分数、NMS/KNN和几何停止梯度；不直接通过几何分支优化预测框，但特征联合训练仍可能间接改变未来检测结果。

“不回写特征”仅指前向中不原位修改候选库，不是切断所有前级梯度。正确性统计头仍是独立detach分支，仅观察当前新头选出的Top-1，不干预排序；其概率应在新模型上重新校准评估。

## 8 维度、计算与实现边界

| 对象 | 默认形状 | 目的 |
|---|---|---|
| 候选/框/检测分类 | B×N×256 / B×N×4 / B×N×2 | 同索引原检测输出 |
| 邻居索引及有效mask | B×N×6 | 只组织上下文，不删目标候选 |
| 相对几何 | B×N×6×5 | 绑定邻居语义与位置 |
| 配对表示 | 概念B×N×6×256 | 实际逐chunk生成 |
| 完整文本 | B×L×256 | 最大长度通常64 |
| 配对得分/聚合权重 | B×N×6 | 六个独立配对 |
| 最终logit/概率 | B×N×1 | 全部候选参与排名 |
| 聚合文本注意力 | B×N×L | 仅诊断显示 |

上下文选择每batch仅将小量框和检测前景分数传CPU，执行贪心去重与KNN，再把索引、几何传回原设备；不转移候选特征和文本。这样避免每个NMS迭代都做CUDA标量同步，但仍有一次主机同步和O(BN²)几何开销，必须实际测速。

每层文本Key/Value只按B幅图投影一次，并在配对chunk中复用；不把完整文本为BNK组反复线性投影。默认chunk_size=256是**配对数**，不是每图候选数。N=256、K=6时每图1536组配对，不重新运行BEiT-3或检测器。

分块降低单次临时张量，但不消除训练时所有配对的反向图。训练dropout使不同chunk设置可能消耗不同随机数；评估或dropout=0时应保持预测和梯度的数值一致性。不承诺开启后速度/显存与原CRS相同。

## 9 配置、训练与旧模型兼容

两份配置`configs/aerialvg/PropVG-aerialvg.py`和`configs/aerialsense/PropVG-aerialsense.py`均增加默认关闭的开关：

```python
enable_relation_scoring = False  # True replaces CRS; missing flag also means False.
relation_neighbors = 6
# model.head 中：
enable_relation_scoring=enable_relation_scoring,
relation_scoring=dict(num_neighbors=relation_neighbors, model_dim=256,
                      geometry_dim=64, num_heads=8, num_layers=2, ffn_dim=512,
                      dropout=0.1, duplicate_iou=0.8, chunk_size=256),
```

| 参数 | 默认 | 含义 |
|---|---:|---|
| num_neighbors | 6 | 每个目标假设的邻居上限，正整数 |
| duplicate_iou | 0.8 | 上下文去重及排除目标近重复框 |
| model_dim / geometry_dim | 256 / 64 | 图文评分维度/几何编码维度 |
| num_layers / num_heads / ffn_dim | 2 / 8 / 512 | 配对文本读取结构 |
| dropout | 0.1 | 新头训练正则，不影响旧头 |
| chunk_size | 256 | 每次处理的配对数 |
| init_seed | 3717 | 独立CPU初始化种子，构造后恢复原RNG |

仅打开关系头而不打开BCEM也可运行；若研究两个模块组合，单独设`enable_bcem=True`并保持原BCEM参数。Grab可开关，teacher可开关。第一版要求`enable_crv=False`，不把替换评分器与旧反事实重排混在一起；冲突会明确报错。旧CRV代码保留，关系开关关闭时不受影响。

不必改训练入口，可在propvg环境运行：

```bash
python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_relation_scoring=True \
  model.head.relation_scoring.num_neighbors=6 work_dir=work_dir/aerialvg/PropVG-relation
```

CLI覆盖要改展开后的`model.head.*`，只覆盖顶层`relation_neighbors`不会自动重算配置中的引用。原多卡脚本也可使用：在它实际读取的配置里改开关和邻居数即可，脚本本身无需重写。

从头训练新实验保持原BEiT-3预训练设置，不加载PropVG整模型。从已有基线初始化则用`--finetune-from /原基线/top1_best.pth`，并清空冲突的resume/load；新头需要训练，不能把随机初始化当作已经训练的替代模型。

```bash
python tools/eval_aerialvg.py /新实验目录/保存的配置.py \
  --checkpoint /新实验目录/top1_best.pth --split test
```

关闭开关时不分配新头、不产生新损失、不改变旧参数名称和初始化；旧配置缺少字段照常走CRS。开启时旧CRS用meta描述器重放历史初始化随机数，不保留其无用权重；其他原模块的初始化仍保持一致，新头追加独立初始化。

关系头注册schema和`trained_steps`。原训练循环在optimizer.step后标记训练步数，随完整checkpoint保存。测试会检查权重完整性、形状、版本及训练状态；旧基线误开新头会报错，新关系模型误关开关也会报错，不能静默换成未训练的CRS。新模型测试应使用原保存配置，K、去重阈值等改变会改变实验设置，即使权重形状不变。

## 10 诊断、验证与消融

`extra_dict`新增：`relation_neighbor_indices`、`relation_neighbor_valid`、`relation_geometry`、`relation_context_kept`、`relation_candidate_valid`、`relation_fallback`、`relation_effective_mask`、`relation_pair_logits`、`relation_weights`，均detach仅供诊断。`attn_map_query`保存以同一邻居权重聚合的最后一层文本注意力，不伪装成原CRS的三层注意力。

检查命令：

```bash
PYTHONWARNINGS=ignore python -m unittest discover -s tools -p 'test_relation_scoring.py'
```

测试覆盖上下文去重与保留所有目标、6近邻、方向/尺度/padding、缺少邻居、空文本/空候选、mask类型、权重聚合、邻居置换不变性、候选置换等变性、分块输出/梯度、文本投影复用、真实256候选维度、特征联合梯度及框坐标截断、完整REFHead、Grab/BCEM/正确性头组合、旧分支与随机数兼容、训练状态及checkpoint误用拒绝。

### 10.1 本次实际验证记录（2026-11-02）

- 新增关系头测试19项，加上可选管线、正确性统计头、BCEM、Grab、旧CRV、checkpoint保留、逐epoch测试及实验记录的相关回归，共174项CPU测试通过。这是选定的相关回归集合，不表示执行了仓库所有测试。
- 另外从修改前的Git HEAD读取原`REFHead`并与当前关闭关系头的实现直接比较：在正确性统计头关闭和开启两种设置下，原权重、随机数状态、前向预测、损失、梯度及评估输出逐项完全一致。
- BCEM现有回归测试新增关闭独立统计头与关系头的显式设置，防止测试因用户当前配置启用尚未训练的统计头而误报；未放宽生产模型的加载检查。
- 当前环境CUDA不可用，尚未进行真实数据GPU训练、AMP或多卡速度/显存验证，也没有精度提升结论；用户原有实验、数据和checkpoint未因验证而改写。

优先比较：背景增强后的原CRS、仅当前候选的评分头、无几何配对、完整配对、打乱邻居特征与几何对应、自适应聚合对比硬max。仅更换邻居排列不应改变结果；打乱语义与位置的绑定则是不同实验。

## 11 需要验证的假设与局限

1. 六近邻不能保证覆盖文本参考物；远处关系、重复框漏去重及背景候选仍可能造成失败。
2. 独立成对读取加权平均没有显式联合证明多个参考条件同时成立；不要把它解释为完整场景逻辑推理。
3. refer监督只给目标身份，不给邻居关系标签，网络可能忽略邻居/几何、退化为自身匹配，或者学习大小位置捷径。需要对应消融，而不只展示权重图。
4. 权重可能过早集中、得分与采用系数可能相互适应；本版先不增加额外约束，观察训练及多随机种子结果再决定。
5. 加权logit不超过所有配对logit的范围；不能仅靠聚合从完全没有正确证据的配对中创造可靠判断。
6. 空邻域回退很少出现时，其输入分布可能训练不足；需统计fallback比例。
7. 模块不依赖teacher不代表teacher对检测覆盖无影响；对照必须统一teacher及检测权重、候选数、BCEM、数据划分、图像尺寸和训练周期。
8. 旧CRS文本pooler的历史实现本次不修改；关闭新头时保持原行为，而新头使用显式bool padding mask。新头收益不能在没有更细对照时全部归因于几何关系。
9. 本实现完成不等于论文创新成立或性能提升；需实测精度、耗时、显存和配对绑定消融。验证范围及限制应以实际执行记录为准，不宣称未经运行的GPU/AMP/多卡结论。
