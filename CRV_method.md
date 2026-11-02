# 不确定性路由与纠错选择器驱动的反事实候选验证（CRV）

当前版本：`schema_version=10`，2026-10-21。

核心文件：`propvg/models/heads/counterfactual_refinement.py`；接入位置为`REFHead.py`中CRS评分之后。当前配置采用最大视觉特征图、全图均值干预、48×48整图评分网格，**不做局部裁剪**。

## 1 动机：不确定不等于应该相信CRV

**目的：** 将“什么时候需要额外验证”与“验证结果是否值得采用”分开学习。

原CRS按候选与完整文本的匹配评分。外观相似目标可能难以区分，CRV通过删除候选区域的视觉差异，检查完整场景—文本支持下降，以提供互补证据。

v9仅用熵产生alpha：`sigmoid(w*U+b)`。20261020_221905实验中，最佳模型的激活子集若直接用CRV排序，会改对210条、改错271条；激活alpha约0.0078，最终首选框变化0条。说明仅知道CRS不确定，不能知道CRV是否更可靠；同时反事实分数乘上小alpha后，通常不足以改变CRS原logit间隔。

本版保留原二维干预和独立评分器，修改三点：

1. 熵只控制是否运行CRV，不再直接预测信任权重。
2. 选择器比较CRS首选与CRV首选，读取候选特征、完整文本和两路分差；用“改对/改错”直接监督。
3. 先把CRV分数校准到CRS logit尺度，再做可学习插值，赋予alpha明确端点含义。

不增加图结构，不改变数据划分、teacher、辅助GT、Grab、BCEM、分割或原定位损失。`enable_crv=False`不构建CRV及选择器，原PropVG管线不变。本版没有开始正式数据集训练，不承诺性能提升。

## 2 整体数据流及张量维度

**目的：** 明确每一步输入、输出、候选对应关系与梯度边界。

记batch为B，候选数N，文本长度L，激活图像数A，重排候选数K=5；原特征通道C=256，评分通道D=128，整图评分网格G=48。

```text
图像 + 完整文本
 → BEiT-3 → 原512通道SimFPN → 检测侧256通道特征 → Deformable DETR
 → 可选Grab → refer_adapter → 可选BCEM → 候选特征Q → CRS → 原logit z
                                                    │
      最大视觉V、完整文本T、预测框、Q、z全部detach ────┘
 → 未达start_epoch：原CRS输出，不训练CRV
 → 达到start_epoch：原高分候选IoU分组 → 归一化熵U
      U≤阈值/图文无效：训练、推理都跳过CRV
      U>阈值：
       ├─完整原场景池化48×48 → 独立图文评分器 → 原支持s0（每图一次）
       └─各Top-K候选二维均值干预 → 完整干预场景池化48×48
                                    → 同一评分器 → 干预支持s_i^-
       d_i=s0-s_i^-；训练另加精确refer GT干预辅助，不加入预测候选
       i=CRS首选，j=CRV下降量首选
       两路一致/反事实分数近乎常量：alpha=0，保留原分数
       否则：
         Q_i,Q_j、完整文本汇总、两路相对分差 → 纠错选择器 → alpha
         原CRS logit + alpha×(校准CRV logit−原CRS logit)
       → 仅原Top-K内部重排，其他候选顺序不变
```

| 张量或操作 | 维度 | 目的 |
|---|---|---|
| 最大检测视觉V | B×256×Hf×Wf，常见144×144 | 保留整图上下文，不重跑编码器 |
| 完整文本T | B×L×256，L通常64 | 不人工拆分目标词/属性词/关系词 |
| 用于CRS的候选特征Q | B×N×256 | 从同一组实际评分候选读取竞争者特征 |
| 预测框、原logit | B×N×4 / B×N×1 | 保留原候选与基础排序 |
| 激活视觉投影 | A×128×Hf×Wf | 无bias 1×1投影，每图一次 |
| 原完整场景池化 | A×128×48×48 | 每图原场景只评分一次 |
| 干预完整场景池化 | AK×128×48×48 | 不截取48×48局部窗口 |
| 精确GT干预 | 每个有GT的激活图额外1份128×48×48 | 训练辅助，推理不读取 |
| 评分器视觉/文本token | S×2304×128 / S×L×128 | S≤chunk_size=8，场景间注意力独立 |
| 原支持、下降量 | 每图1个 / 每图K个 | 反事实独立候选评分 |
| 竞争候选特征 | 2×256 → 2×64 | 选择器只比较两路首选 |
| 完整文本汇总 | L×256 → 256 → 64 | 为选择器提供表达式语义 |
| 相对分数统计 | 4维 | 比较两路竞争强弱，不以熵代替信任 |
| 选择器输入/隐藏/输出 | 260 → 128 → 1 | 轻量预测采用CRV的程度 |
| 校准/融合logit | 每图K维 | 尺度匹配后重排 |
| 最终排序优先级 | B×N×1 | 保留所有候选，仅Top-K内部交换 |

Q严格取`refer_query_embed`，即经过已有Grab/BCEM配置处理后、实际送入CRS的特征，与bound_boxes、CRS分数共用候选索引。不是从其他解码层或未筛选的query中错位取特征。

## 3 熵路由、启训epoch与去重

**目的：** 将额外计算集中在原模型有歧义的样本，不干预明确预测。

原CRS前Nu=5个候选按分数顺序贪心分组：与已保留代表框的IoU≥0.8则视为重复，仅从熵统计中排除，不删除预测框。对m个代表logit：

$$p_j=\operatorname{softmax}_j(z_j/\tau_u),\qquad
U=-\frac{\sum_jp_j\log p_j}{\log m},\quad
r=\mathbb1[\text{图文有效}]\mathbb1[U>u_0].\tag{1}$$

m<2时U=0，tau_u=1。阈值相等时也跳过。训练和推理均先路由，r=0不构建mask、不投影视觉、不运行评分器或选择器，保留原CRS概率与排序。

`crv_start_epoch`为1-based。启训前不计算熵，所有CRV损失为常数0，CRV梯度为None，AdamW不因weight decay更新它。达到epoch后在同一次训练中联合优化，不重建优化器，不冻结CRS，不增加分阶段训练。最近训练epoch保存在checkpoint，独立测试据此判断是否启用。

## 4 最大视觉特征图上的二维均值干预

### 4.1 框坐标与padding

**目的：** 避免干预区域和预测框错位。

框在有效图像坐标系归一化，cxcywh转xyxy并裁剪到[0,1]。有效图像(H,W)、Pad画布(Hp,Wp)、特征(Hf,Wf)下，格点位置为：

$$x_{uv}=\frac{(u+.5)W_p/W_f}{W},\qquad
y_{uv}=\frac{(v+.5)H_p/H_f}{H}.\tag{2}$$

### 4.2 扩张和羽化

**目的：** 覆盖目标边缘，避免硬切边；不再受固定局部窗口截断。

候选中心c、半宽高a，rho=max(|x−cx|/ax,|y−cy|/ay)，t=clip((rho−1.5)/0.5,0,1)：

$$M_i(u,v)=\tfrac12(1+\cos(\pi t))A(u,v).\tag{3}$$

1.5倍框内完全替换，1.5～2倍羽化，2倍外不变；A排除padding。零面积框mask为0。mask在原最大特征分辨率上生成，不先在48×48评分网格上画。

### 4.3 每通道的整图均值替换

**目的：** 抑制候选视觉差异而非置零，不混合不同通道。

$$\mu_c=\frac{\sum_{uv}A_{uv}V_{cuv}}{\sum_{uv}A_{uv}},\qquad
\widetilde V_{i,cuv}=V_{cuv}+M_i(u,v)(\mu_c-V_{cuv}).\tag{4}$$

均值来自整幅有效图像，不来自局部窗口。无bias线性1×1投影与均值替换可交换，因此先投影256→128，每图缓存一次均值。

### 4.4 全图池化和分块

**目的：** 保留远处上下文，同时控制token数及临时显存。

投影特征P按有效覆盖率池化到G×G，当前G=48：

$$V^0=\Pi_A(P),\qquad
V_i^-=V^0+\frac{\mu\operatorname{AvgPool}_G(M_i)-\operatorname{AvgPool}_G(PM_i)}
{\operatorname{AvgPool}_G(A)}.\tag{5}$$

这是完整高分辨率均值干预后再整图有效池化的线性等价实现，支持非整除尺寸；无效格置0。原场景池化每图一次，加权干预按chunk临时计算，不同时保存所有完整高分辨率副本。

默认训练每图1+K+1=7份场景，最后一份是GT辅助；推理每图1+K=6份。跨图像/候选批处理，但不在token维拼接不同场景。缓存文本投影和位置编码只在当前前向复用，保留CRV内部梯度，不跨训练步缓存旧参数结果。

## 5 独立图文支持评分

**目的：** 衡量删除候选前后的场景—完整描述匹配变化，不复用CRS评分器。

2304个128维视觉token加整图二维位置编码；完整文本256→128后加一维正余弦编码。文本为Query、视觉为Key/Value，经过2层Transformer Decoder，每层文本self-attention、图文cross-attention、512维FFN，4头，dropout=0。文本和视觉padding不参与有效读取。

文本交互前/后均值t、h及视觉均值v拼接为512维，MLP为512→128→1，GELU，末层无bias：

$$s^0=S_\theta(V^0,T),\qquad s_i^-=S_\theta(V_i^-,T),\qquad d_i=s^0-s_i^-.\tag{6}$$

支持是未校准实值，不是文本成立概率；绝对support=1不表示100%匹配。d可为负。零mask时强制支持差严格为0，排除不同批形状GEMM的微小数值误差。

同图s0是公共标量，在候选相对排序中会抵消；原分支仍通过下降量锚定及精确GT辅助得到梯度，不能声称公共s0本身提供候选间相对判别。

## 6 纠错选择器：权重从哪里学

### 6.1 找到竞争双方

**目的：** 具体判断CRV是否能纠正当前首选，而不是仅判断样本难不难。

在原Top-K按CRS排序的集合内，i=0为CRS首选，j=argmax(d)为CRV首选；CRV并列时沿原CRS顺序稳定打破平局。不通过GT选择候选。

若i=j则不需要纠正，alpha=0；若std(d)≤1e-4则反事实差异不足，alpha=0，避免归一化放大近乎常量的噪声。此时CRV辅助损失仍可训练评分器，不误当成低熵整分支跳过。

### 6.2 门控输入、结构和梯度

**目的：** 读取两路意见冲突的证据，又不允许门控改变上游或操纵评分器来伪造置信度。

两竞争候选Q_i、Q_j分别经共享LayerNorm(256)→Linear(256,64)→GELU。完整文本先按有效token均值汇总，再经独立的同结构投影到64维。

四个标量为：CRS对i/j的分差、CRV对j/i的下降量差、CRS前两名分差、CRV前两名分差，分别按本路尺度归一化。**不把熵输入门控；熵仅负责步骤3的路由。**

$$x_g=[P_q(Q_i),P_q(Q_j),P_q(Q_j)-P_q(Q_i),P_t(\bar T),\Delta z_{ij},\Delta d_{ji},\Delta z_{12},\Delta d_{12}],\tag{7}$$
$$a=g_\phi(x_g),\quad p_g=\sigma(a),\quad
\alpha=r\,\mathbb1[i\ne j]\mathbb1[\sigma_d>\epsilon]p_g.\tag{8}$$

输入260维，经Linear(260,128)、GELU、Linear(128,1)。候选/文本源特征和全部分数统计detach；选择器自己的投影及MLP正常训练。输出层weight初始化0，bias=log(.02/.98)，初始可用分歧的p_g=.02，不强制alpha下限。该初始化下第一步主要更新输出层，之后内部表示逐步获得梯度。

### 6.3 训练标签：明确区分纠错和伤害

**目的：** 直接学习“该不该信CRV”，而不是依靠最终排序损失间接缩小alpha。

用原始refer GT，以IoU≥0.5判定两路首选是否正确：

| CRS首选i | CRV首选j | 选择器标签 |
|---|---|---|
| 错 | 对 | y=1，鼓励信CRV |
| 对 | 错 | y=0，鼓励保留CRS |
| 对 | 对 | 忽略门控分类监督 |
| 错 | 错 | 忽略门控分类监督 |

还要求熵激活、i≠j、反事实分数非退化。GT只生成训练标签，不进入选择器输入。两路都对/都错不当作负标签，以免大量无效样本把门控压成关闭。**忽略的只是新增分类项**，原CRV辅助与最终排序监督仍按原规则计算。

对有效正、负类别分别求BCE均值，再对当前有样本的类别平均：

$$L_{gate}=\frac1{|\mathcal C|}\sum_{c\in\mathcal C}
\frac1{N_c}\sum_{b:\,y_b=c,\,valid_b}\operatorname{BCEWithLogits}(a_b,y_b).\tag{9}$$

两类都有时各占一半；只出现一类时只求该类均值；无有效标签时为0。多卡的N_c在全局batch统计，不能各卡独立平衡后简单平均。实现每个达到启训epoch的训练进程固定一次3标量all_reduce（两类计数及预测正确数），包括没有激活样本/GT标签的进程；梯度按DDP的平均约定乘world_size，得到正确的全局类别均衡损失。不新增大张量通信。

门控分类loss不乘alpha，不向独立评分器、CRS或主干反传。即使alpha很小，选择器仍有直接的纠错梯度。该分类标签判断两个端点，不保证中间融合一定正确，故仍保留最终定位监督。

## 7 分数校准与最终融合

**目的：** 让权重表达信任，而不是被两路数值尺度差异限制；保证alpha=1确实能采用CRV排序。

原Top-K logit z全部detach。定义总体标准差（unbiased=False）：

$$\mu_z=\operatorname{mean}(z),\quad
\sigma_z=\max(\operatorname{std}(z),0.1),\quad
\sigma_d=\operatorname{sg}(\operatorname{std}(d)),\tag{10}$$
$$\widehat d_i=\mu_z+\sigma_z\frac{d_i-\operatorname{mean}(d)}{\max(\sigma_d,10^{-4})},\qquad
z_i^f=z_i+\alpha(\widehat d_i-z_i),\quad p_i^f=\sigma(z_i^f).\tag{11}$$

均值匹配保留logit中心，尺度匹配避免小量纲下降无法竞争；0.1下限避免原CRS完全打平时校准分数也退化。反事实标准差作为归一化分母detach，均值中心化仍对d可导。近常量分数由上一节门控条件明确回退，不放大噪声。

- alpha=0：严格保留原CRS logit，回退时原sigmoid概率也直接保留。
- alpha=1且分支有信息：得到校准CRV排序，校准不改变d的排序。
- 中间值：在同一logit尺度上融合，仍通过最终GT排序loss优化。

这是**校准logit插值**，不是softmax概率混合，也不再是v9的`z+alpha*d`。这项变化是为了让“提高门控信任”的标签有可实现的端点。选择器围绕两个首选决策，但融合对Top-K全部候选执行，中间权重可能选出第三个候选，必须依靠最终排序监督和实际纠错统计检验。

只在原Top-K内重排，其余候选顺序不变，全部N个框保留。默认K=5时不改变Top-5命中集合。后处理沿用ranking_priority/ranking_logits，不按更新后的sigmoid在全部N个框中重新取Top-K。

## 8 所有损失及参数流向

**目的：** 说明哪些监督训练评分器、哪些训练选择器，避免新增loss漏入总损失或影响原管线。

### 8.1 原反事实辅助项：定义和权重不变

锚定正样本IoU>0.5，可靠负样本IoU=0且完整扩张mask对GT覆盖≤.01，其他不参与锚定。覆盖采用特征格与GT的分数相交面积，避免小目标漏算：

$$L_{anchor}=\operatorname{mean}_{p}[1-d_p]_+
+\operatorname{mean}_{n}\operatorname{SmoothL1}(d_n,0).\tag{12}$$

独立排序以IoU≥0.5为正集合P_loc；没有正确候选时跳过，允许多正框：

$$L_{rank}^{cf}=\operatorname{logsumexp}_i(d_i/\tau_{cf})
-\operatorname{logsumexp}_{i\in P_{loc}}(d_i/\tau_{cf}),\qquad
L_{cf}=L_{anchor}+L_{rank}^{cf}+.1[1-d_{GT}]_+.\tag{13}$$

tau_cf=1仅用于该项及诊断，不进入融合；精确GT干预不进入预测候选，不被teacher框替代。原支持分没有绝对BCE标签。

### 8.2 原最终定位及防改错项：使用新融合logit

以IoU≥0.5作为最终排序正集合P，其余为负。无正或无负时loss为0：

$$M_0=\operatorname{sg}(\max_pz_p-\max_nz_n),\quad
M_f=\max_pz_p^f-\max_nz_n^f,\tag{14}$$
$$L_{fusion}=w_{repair}\operatorname{softplus}(.5-M_f)
+2\mathbb1[\text{原CRS正确}][\min(M_0,.5)-M_f]_+.\tag{15}$$

原CRS错误时w_repair=2，否则1。安全项约束间隔，不能提供逐样本不改错保证。

### 8.3 总损失及归一化

$$L=L_{base}+.1\operatorname{mean}_{B}(rL_{cf})
+1.0\operatorname{mean}_{B}(rL_{fusion})+1.0L_{gate}.\tag{16}$$

前两项CRV损失按完整本卡batch平均，未激活样本贡献0；选择器按全局有效类别均值归一化，**不是**简单除以B，避免有效纠错样本被大量双方都错样本稀释。日志字段均已乘外层权重：

| 字段 | 外层权重 | 更新参数 |
|---|---:|---|
| loss_crv_cf | 0.1 | 独立图文支持评分器 |
| loss_crv_fusion | 1.0 | 评分器及纠错选择器 |
| loss_crv_gate | 1.0 | 选择器的候选/文本投影和MLP，不更新评分器 |
| 原loss_det/loss_refer等 | 原值 | 原检测、CRS、主干及原配置模块 |

新增loss已接入训练总loss和日志/SwanLab，不是只计算显示。入口全部detach，选择器统计再次detach；CRV与基础模型独立梯度裁剪，初始化使用独立CPU种子1717并恢复RNG。没有冻结原CRS，也没有让新loss改其梯度。

## 9 配置、日志与兼容性

**目的：** 允许按原命令运行，明确新参数和诊断含义。

两份数据集配置保留现有CRV开关、K、G、启训epoch、阈值、数据和其他模块开关，仅补充新门控参数：

```python
enable_crv = True  # 示例；实际开关以当前配置为准
crv_topk = 5
crv_grid_size = 48  # 全图池化，不是裁剪
# model.head.crv 中：
gate_loss_weight = 1.0       # 全局类别平衡的纠错选择监督
gate_dim = 64               # 候选与文本投影维度，MLP隐藏维为128
calibration_eps = 1e-4      # 近常量CRV分数回退阈值/分母下限
calibration_min_scale = 0.1 # CRS标准差下限
gate_initial_probability = 0.02
```

原命令不变，例如：

```bash
python tools/train.py configs/aerialvg/PropVG-aerialvg.py
```

CLI覆盖需修改展开后的`model.head.enable_crv`或`model.head.crv.*`，不是只改顶层引用变量。数据集配置显式传G=48；省略grid_size的构造调用仍保留默认24。显存紧张可减小chunk_size，语义不变。不接受旧数值crop_size，以免误开局部裁剪。

原alpha/active/uncertainty/corrected/harmed等日志保留，新增：

- `crv_disagreement`：激活且两路首选不同的比例，占完整batch。
- `crv_gate_positive/negative`：有效纠错正/负标签比例，占完整batch。
- `crv_gate_accuracy`：有效标签上的原始门控概率0.5阈值分类准确率，多卡按全局有效标签数统计；无标签时0为占位。
- `crv_gate_probability`：可用分歧样本上的平均门控概率，区别于包含跳过/一致样本0值的crv_alpha；没有可用分歧时0为占位。多卡日志按各卡该均值平均，不是严格按分歧数量加权。
- `crv_calibration_scale`：CRS尺度/CRV尺度的比例，按完整batch平均，供尺度诊断，不等于信任权重。

单样本extra还保存原/干预支持、原/校准/融合logit、门控概率、是否有监督、训练标签及竞争候选原始ID。推理的gate_supervised=False，不使用GT生成标签。

schema升为10，旧v1～v9 CRV权重与新门控不兼容，不能直接load/resume当作本版模型；旧实验用自身备份代码复现。从不含CRV的基线初始化可按原`--finetune-from`流程，清空冲突的resume/load设置；同版本配置checkpoint可续训。关闭CRV时旧基线不受此检查影响。

多卡沿用`find_unused_parameters=True`，支持启训前跳过、部分卡无有效标签和全部跳过后再启用。推理不执行门控损失计数通信。`start_test`行为、checkpoint保存与目录命名逻辑均不改。

## 10 验证及局限

**目的：** 区分实现正确性、选择器可学习性与真实定位收益。

回归覆盖真实256×144×144视觉、64文本token、128评分通道和48×48评分网格；候选特征对齐、上游梯度隔离、校准端点、近常量回退、两路一致保真、零mask保真、Top-K边界、直接门控标签、类别平衡BCE、训练总loss和日志接入。合成样本测试可在相同熵下学习不同门控权重，但不等价于真实数据泛化证明。

本次实际验证：44项CRV测试及86项原Prop/Grab/BCEM、checkpoint、逐轮测试和实验日志相关回归通过，共130项。CPU/Gloo两进程6步检查通过；另在物理2、3卡上以256×144×144视觉、64文本token、128评分通道和48×48网格进行CUDA/NCCL两进程FP32合成检查，覆盖全局类别平衡梯度与准确率、部分进程无标签、全部跳过、重新启用、AdamW不误更新及checkpoint启训状态恢复，均通过。编译和`git diff --check`通过。未启动正式数据集训练，不将这些正确性/合成检查视为真实性能提升或AMP验证。

局限：

1. CRV自身若候选判别弱，选择器只能筛选，不能凭空生成正确框；Top-K没有正确框时无法修复。
2. 中间插值可能选出非两路首选的第三候选；最终排序监督不能保证完全不改错。
3. 类别均衡BCE改变有效训练类别的相对权重，门控概率不是自然数据分布下校准好的“改对概率”，也不是严格最优alpha。
4. 标准化可能强化不可靠的排序；近常量回退只处理退化输入，不保证其余分数都可靠。
5. 高层图文特征mask不是真实删除物体，语义残留、参考物体误干预及小目标池化稀释仍存在。

本次不声称已经获得性能提升。需通过真实训练后同一checkpoint的CRS、CRV、融合三路，以及改对/改错、有效门控分类表现验证。
