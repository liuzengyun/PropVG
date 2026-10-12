# 面向航拍视觉指代的文本稀疏背景选择与非线性候选增强方法

**模块名称：** 背景上下文增强模块（Background Context Enhancement Module，BCEM）。

**方法概述：** 在最高分辨率检测特征图上，围绕每个预测候选读取八个框外扇区，每个扇区池化为一个背景向量。扇区表示仅由视觉特征经共享 MLP 与 LayerNorm 得到；以原始指代表达的上下文化 token 为 Query、扇区视觉表示为 Key，计算文本与区域内容的相关性。有效 token 的注意力平均后，通过 sparsemax 得到含有严格零值的扇区汇总权重，仅用被选中的扇区 Value 构造背景证据。随后对齐背景尺度，用可学习门控控制背景输入强度，将原候选与门控背景拼接，通过小型 MLP 学习联合表示，送入 CRS 完成指代评分。选择数量由文本和扇区内容共同决定，可为一个、两个或全部八个，不采用固定 top-k。

**实现状态：** 本文对应当前图像边界自适应扇区、文本稀疏背景选择与拼接 MLP 融合版。默认关闭显式位置编码，不向表达添加 top/right 等方向提示，不向区域特征添加方向或相对坐标编码。每条扇区射线从候选框边界延伸到有效图像边缘，其长度由候选位置、框大小和图像尺寸共同确定，不再设置固定短边比例或 sector_radius 参数。每扇区默认使用 5×5 个固定点近似区域池化，尚不是完整扇区特征网格的掩码池化。逐 token 注意力沿有效扇区归一化，最终扇区权重采用 sparsemax 稀疏选择，候选与门控背景经拼接 MLP 输出；不再使用候选与背景的加权相加、文本乘法增益或额外方向评分 MLP，不保留四尺度证据序列、相邻转移、GRU 或多层交互编码器。本文件位于仓库根目录，参数直接配置在现有 AerialVG 配置中。

## 1 研究背景、动机与科学问题

### 1.1 航拍目标的外观歧义与背景补充需求

航拍图像中，车辆、行人等目标通常具有尺寸较小、同类外观相似和分布密集的特点。指代表达中的颜色、类别等外观属性往往只能筛出一组候选，难以独立确定所指目标。道路、停车场、建筑等场景环境，以及周围车辆或行人等辅助对象，则提供了进一步消歧的依据。例如，“停车场内的白色轿车，上方有一辆 SUV”同时包含目标属性、场景环境和辅助对象关系。

AerialVG 将相似目标之间的空间关系作为任务的重要挑战，其论文还指出，道路等非结构化背景信息仍难以被现有模型充分利用。[AerialVG 原始论文](https://arxiv.org/html/2504.07836v3)

因此，有必要研究一种候选相关的背景表示，使评分模型能够在目标自身线索之外，利用候选周围的场景与辅助对象证据。

### 1.2 现有模型的表示缺口

PropVG 已通过 BEiT-3 完成图文编码与融合，候选查询还与多尺度视觉特征交互，因此其候选表示已具备隐式上下文信息。本方法的动机不是“基线没有背景”，而是补充一个显式围绕候选组织和使用背景的分支。[PropVG 原始论文](https://arxiv.org/html/2509.04833v1)

整体区域池化可能将上方建筑、右侧道路等信息混合，弱化它们相对于候选的位置差别。另一方面，不同表达对背景的依赖程度不同：颜色和类别可能主要依靠候选表示，而“上方建筑旁的轿车”还需要方向相关的环境证据。固定融合比例无法随候选及表达调整背景用量。为此，本研究以八个方向扇区为基本单元，分别学习“背景内部应关注哪些方向”和“候选与整体背景如何融合”，避免以复杂的关系序列建模增加实现负担。

候选表示主要承载目标身份与外观，区域背景主要承载环境及辅助对象，两者维度相同并不意味着逐通道语义相同。直接以互补权重逐维混合还会在增大背景权重时同步减小候选项。本研究进一步采用候选与背景拼接的非线性融合，在输入端分别保留两路表示，让共享 MLP 学习它们的组合关系。该设计提供更灵活的组合能力，但不能仅凭结构认定其优于线性混合。

仅采用普通 softmax 时，每个有效扇区通常仍得到正权重，因而“重要性较低”并不等于“未参与汇总”。本研究进一步区分相关性估计与最终区域使用，通过可微的稀疏单纯形投影将部分权重精确置零，同时允许多个方向共同提供证据。其目标是减少无关背景混入，而非预先规定只使用少数方向；所有方向同等相关时，应允许全部保留。

这里的“背景”指候选框外上下文，可以包含其他前景对象；其特征也可能通过骨干感受野混入候选信息，不能称为严格分离的纯背景。

### 1.3 拟解决的关键问题

本方法聚焦三个相互衔接的问题：如何以少量区域表示覆盖候选周围环境；如何让原始文本依据视觉内容选择相关扇区；如何在特征尺度可比的条件下，控制背景输入强度并学习候选与背景的联合表示。

对应的机制假设为：八扇区分区能够避免在文本选择之前将全部背景混合；文本与区域视觉内容之间的相关性能够筛选背景证据；候选级门控能够适应不同表达对背景的需求。当前默认版不将显式方向对齐作为机制或贡献。上述假设必须通过独立训练的基线、均匀扇区读取及固定背景输入门控等对照验证，不能提前表述为已取得的性能提升。

## 2 总体结构与输入定义

设有效图像尺寸为 $H\times W$，padding 画布尺寸为 $\widetilde H\times\widetilde W$。用于射线边界求交的图像区域与短边尺度定义为：

$$
\Omega=[0,W]\times[0,H],\qquad S=\min(H,W).
\tag{1}
$$

射线覆盖由有效图像区域 $\Omega$ 决定，不向 padding 延伸。短边尺度 $S$ 仅供可选位置编码的相对偏移归一化使用，不控制默认扇区的长度。框外扇区在连续几何上覆盖到图像边界，实际插值点使用区域内部的积分单元中心，不要求点本身落在最外侧边界上。

候选 $i$ 的像素坐标框为 $b_i=(c_i^x,c_i^y,w_i,h_i)$，中心记为 $\boldsymbol c_i$。原候选表示为 $q_i\in\mathbb R^C$，文本特征为 $T=[t_1,\ldots,t_L]$，最高分辨率检测侧特征为 $F\in\mathbb R^{C\times H_f\times W_f}$。首版采用 $C=256$、背景维度 $D=128$。

模块总体映射为：

$$
(F,b_i,q_i,T)\longmapsto
\{f_{i,d}\}_{d=1}^{8}\longmapsto\{A_{i,\ell,d},h_{i,\ell}\}_{\ell=1}^{L}
\longmapsto \{\alpha_{i,d}\}_{d=1}^{8}\longmapsto e_i
\longmapsto (b_i^{ctx},g_i)\longmapsto q_i^{enh}
\longmapsto s_i.
\tag{2}
$$

其中 $f_{i,d}$ 是扇区视觉背景特征，$A_{i,\ell,d}$ 是文本 token 对扇区的读取注意力，$h_{i,\ell}$ 是对应的概念性视觉读取，$\alpha_{i,d}$ 是最终稀疏扇区权重，$e_i$ 是仅由被选扇区构成的汇总背景，$b_i^{ctx}$ 是对齐到候选空间与尺度的背景表示，$g_i$ 是背景输入门控，$s_i$ 由原 CRS 产生。候选几何框决定背景采样，原候选表示参与最终门控与拼接融合，不直接输入扇区注意力。$b_i^{ctx}$ 为特征，区别于候选几何框 $b_i$。

## 3 八个框外扇区的构建与池化

### 3.1 扇区布局与覆盖范围

图像横轴向右、纵轴向下，八个主方向角依次为：

$$
\Theta=
\left\{-\frac{\pi}{2},\frac{\pi}{2},\pi,0,
-\frac{3\pi}{4},-\frac{\pi}{4},
\frac{3\pi}{4},\frac{\pi}{4}\right\}.
\tag{3}
$$

它们对应上、下、左、右、左上、右上、左下、右下。每个扇区覆盖主方向两侧各 $22.5^\circ$，八个扇区在角度上构成完整分区。

每条射线先从候选中心到达候选框边界，再继续延伸至其与有效图像边缘的交点。因此区域是被候选框截去内部、被图像边界截断的扇形状区域，外边界随射线变化，不是固定半径的圆弧。候选偏左时向右覆盖较长，靠近上边缘时向上覆盖较短；矩形图像和斜方向同样按真实边界求交。

例如图像为 $100\times200$ 像素、候选中心为 $(100,50)$、框宽高为 $(40,20)$ 时，正上方从 $y=40$ 延伸到 $y=0$，长度为 40；正右方从 $x=120$ 延伸到 $x=200$，长度为 80。两个方向的长度由几何决定，不共享固定覆盖值。若某条射线的框边界已达到或超过图像边缘，该射线没有可用框外区域。

### 3.2 固定网格近似区域读取

每个扇区使用 $A=5$ 个角度单元和 $K=5$ 个径向单元，在单元中心取点：

$$
\delta_a=
\left(\frac{a+\tfrac12}{A}-\frac12\right)\frac{\pi}{4},
\qquad
\zeta_k=\frac{k+\tfrac12}{K},
\quad
a=0,\ldots,A-1,\ k=0,\ldots,K-1.
\tag{4}
$$

默认角度偏移为 −18°、−9°、0°、9°、18°。径向比例为 0.1、0.3、0.5、0.7、0.9，在每条射线自己的完整框外区间内映射到实际位置。径向位置用于近似区域积分，最终全部汇总为一个向量，不形成多尺度证据或距离序列。

令 $\boldsymbol d_{d,a}=[\cos(\theta_d+\delta_a),\sin(\theta_d+\delta_a)]$。候选中心到框边界交点的距离为：

$$
\lambda_{i,d,a}
=\min\left\{
\frac{w_i/2}{\max(|d^x_{d,a}|,\varepsilon)},
\frac{h_i/2}{\max(|d^y_{d,a}|,\varepsilon)}
\right\},\qquad \varepsilon=10^{-7}.
\tag{5}
$$

对于位于有效图像内的候选中心，定义射线到左右边界的距离：

$$
\ell_x=\begin{cases}
(W-c_i^x)/d^x_{d,a}, & d^x_{d,a}>\varepsilon,\\
-c_i^x/d^x_{d,a}, & d^x_{d,a}<-\varepsilon,\\
+\infty, & |d^x_{d,a}|\le\varepsilon.
\end{cases}
\tag{5a}
$$

到上下边界的距离 $\ell_y$ 同理，将 $W,c_i^x,d^x_{d,a}$ 替换为 $H,c_i^y,d^y_{d,a}$。近似平行于某轴边界的射线在该轴取正无穷，不错误地截断另一轴上的覆盖。图像边界交点的中心距离、实际框外长度、采样半径与位置依次为：

$$
\begin{aligned}
\mu_{i,d,a}&=\min(\ell_x,\ell_y),\\
R_{i,d,a}&=\max(\mu_{i,d,a}-\lambda_{i,d,a},0),\\
r_{i,d,a,k}&=\lambda_{i,d,a}+\zeta_k R_{i,d,a},\\
\boldsymbol p_{i,d,a,k}&=\boldsymbol c_i+r_{i,d,a,k}\boldsymbol d_{d,a}.
\end{aligned}
\tag{6}
$$

几何计算采用 float32，并对候选框停止梯度。实现将采样位置转换到 padding 画布对应的网格：

$$
\boldsymbol g(\boldsymbol p)=
\left[
2p^x/\widetilde W-1,\,
2p^y/\widetilde H-1
\right].
\tag{7}
$$

使用双线性插值及 align_corners=False。有效图像尺寸用于定义区域范围与越界条件，padding 尺寸用于坐标映射，二者分别处理。

### 3.3 有效性约束与面积加权池化

对 25 个采样点分别检查：位于有效图像内、位于候选框外、候选宽高为正，并且插值支撑区域不接触视觉 mask 标记的 padding 或网格外部。最后一项通过插值有效性图得到的支撑权重实现，阈值为 $1-10^{-6}$。

记点有效性为 $m_{i,d,a,k}$，有效点数与扇区有效性为：

$$
n_{i,d}=\sum_{a,k}m_{i,d,a,k},
\qquad M_{i,d}=\mathbf1[n_{i,d}\ge5].
\tag{8}
$$

等间隔径向比例不代表等面积区域。极坐标面积元为 $r\,dr\,d\theta$，不同射线的框外长度又各不相同。由于 $r=\lambda+\zeta R$，有 $dr=R\,d\zeta$，因此面积权重必须同时保留半径 $r$ 和射线跨度 $R$：

$$
\omega_{i,d,a,k}
=m_{i,d,a,k}\,r_{i,d,a,k}R_{i,d,a},
\qquad
\bar\omega_{i,d,a,k}
=\frac{\omega_{i,d,a,k}}
{\max(\sum_{a',k'}\omega_{i,d,a',k'},10^{-6})}.
\tag{9}
$$

角度步长与无量纲径向比例步长在各单元间相同，归一化后消去；不同射线的真实径向跨度 $R_{i,d,a}$ 必须保留。若只按采样半径加权，将低估较长射线代表的区域面积。长度为零的射线面积权重也为零。式（9）用于近似面积平均，不增加可学习关系分支。

先在整幅特征图上执行一次无偏置 $1\times1$ 投影 $\widetilde F=W_vF$，将视觉 padding 零化，再计算：

$$
u_{i,d}
=M_{i,d}\operatorname{LN}
\left(
\sum_{a,k}\bar\omega_{i,d,a,k}
\widetilde F(\boldsymbol p_{i,d,a,k})
\right)\in\mathbb R^D.
\tag{10}
$$

LayerNorm 用于稳定区域特征尺度。每个候选只产生八个 $u_{i,d}$。固定 5×5 网格是稀疏近似，不能保证覆盖扇区内所有细小对象；延伸至图像边缘后，远距离方向的点间距增大，近邻细节仍可能遗漏。图像边界附近若插值邻域接触 padding 或特征网格外部，该点继续屏蔽。框外屏蔽约束的是采样位置，骨干感受野与插值邻域仍可能包含候选自身的信息。

## 4 原始文本选择视觉背景与候选—背景权重学习

### 4.1 仅由视觉内容构造扇区表示

对式（10）的扇区视觉使用共享小型 MLP 和无仿射参数的 LayerNorm：

$$
f_{i,d}=\operatorname{LN}\bigl(\phi_v(u_{i,d})\bigr),
\qquad \phi_v:\mathbb R^{128}\rightarrow\mathbb R^{128}.
\tag{11}
$$

MLP 为 $128\rightarrow128\rightarrow128$，中间采用 GELU。八个扇区共享相同的视觉映射、Key 与 Value 投影，不分配方向专属参数，不拼接位置编码，也不添加扇区编号嵌入。原始表达由基线 BEiT-3 编码，其上下文化 token 直接经线性投影用于注意力；不改写表达或额外加入方向词。

方向及候选坐标仍用于划定采样区域和屏蔽无效位置，但不作为默认注意力的附加输入。这里去除的是额外扇区方向身份；输入视觉特征仍可能携带骨干原有的位置和上下文信息，不能称为完全无位置信息。

在相同有效性下，若两个扇区具有相同池化视觉，则其表示相同：

$$
u_{i,d}=u_{i,d'}\quad\Longrightarrow\quad f_{i,d}=f_{i,d'}.
\tag{12}
$$

令 $\Pi$ 表示对扇区视觉及有效性一起进行排列。由于共享映射和内容注意力不使用扇区编号，最终权重按相同顺序排列，汇总背景保持不变：

$$
\boldsymbol\alpha_i^{\Pi}=\Pi\boldsymbol\alpha_i,
\qquad e_i^{\Pi}=e_i.
\tag{13}
$$

因此，当前选择依据是“哪个区域的视觉内容与原始表达相关”，而非显式建立 top 与上方扇区的固定身份关联。对于“上方建筑”和“下方建筑”仅靠方位才能区分的情形，去掉额外方向信息可能限制区分能力，需要实验验证。历史位置分支只保留为 use_position=True 的可选对照，默认不实例化。

### 4.2 文本 Query 读取扇区视觉 Key、Value

令 $m_\ell^{text}=1$ 表示有效 token，保留完整表达的逐 token 特征并投影，同时计算最终融合门控使用的句子特征：

$$
\xi_\ell=W_Q t_\ell,\qquad
t=\frac{\sum_{\ell=1}^{L}m_\ell^{text}\xi_\ell}
{\max(1,\sum_{\ell=1}^{L}m_\ell^{text})}
\in\mathbb R^D.
\tag{14}
$$

文本投影 $W_Q$ 为无偏置 $256\rightarrow128$ 线性层，同时用于注意力 Query 与句子汇总。padding 特征在投影与平均之前零化。当前仓库的文本 mask 为 padding=1，使用时先转换为布尔值再取反。

原始表达的上下文化文本 token 作为 Query，扇区视觉作为 Key、Value。对候选 $i$，每个有效 token 在八个有效扇区之间计算点积注意力并读取视觉证据：

$$
\begin{aligned}
\eta_{i,\ell,d}&=\frac{\xi_\ell^\top W_K f_{i,d}}{\sqrt D},\\
A_{i,\ell,d}&=\operatorname{MaskedSoftmax}_{d}(\eta_{i,\ell,d}),\\
h_{i,\ell}&=\sum_{d=1}^{8} A_{i,\ell,d}W_V f_{i,d}.
\end{aligned}
\tag{15}
$$

其中 $W_K,W_V:\mathbb R^{128}\rightarrow\mathbb R^{128}$ 均为无偏置线性投影。Query 来源于原始表达；Key、Value 来源于扇区视觉。因此读出的 $h_{i,\ell}$ 是文本选择的图像背景证据，不是文本向量，也不再经过文本条件通道乘法增益。注意力权重本身承担扇区重要性分配，不另设方向评分网络。

例如 building、road 或 silver SUV 的上下文化 token 可以根据扇区视觉内容读取相关环境或辅助对象。表达本身包含的 top、right 等词仍按原文保留，不额外注入方向提示或规则，也不预设它们必然选择对应的几何扇区。该内容选择能力需要训练和可视化验证。

注意力仅沿有效扇区归一化：有效 token 且存在有效扇区时，每个 token 的八方向权重和为 1；padding token、无效扇区及全部背景无效的行均为零。实现将无效 logits 置为有限的极小值，再于 softmax 后零化无效位置，避免全无效行产生 NaN。logits 与 softmax 使用 float32，未叠加多头、多层交互或序列编码器。

### 4.3 有效 token 汇总与自适应稀疏扇区选择

定义 $\mathcal V_i=\{d:M_{i,d}=1\}$ 及 $H_T=\mathbf1[\sum_\ell m_\ell^{text}>0]$。先平均有效 token 的注意力，形成整句表达对八个扇区的稠密相关性，再进行稀疏投影并读取 Value：

$$
\begin{aligned}
\bar\alpha_{i,d}&=\frac{\sum_\ell m_\ell^{text}A_{i,\ell,d}}
{\max(1,\sum_\ell m_\ell^{text})},\\
\boldsymbol\alpha_i&=\operatorname{Sparsemax}_{\mathcal V_i}
\left(2\bar{\boldsymbol\alpha}_i\right),\\
e_i&=\sum_{d=1}^{8}\alpha_{i,d}W_V f_{i,d}.
\end{aligned}
\tag{16}
$$

对于有效文本与非空有效扇区集，Sparsemax 是到有效扇区概率单纯形的欧氏投影：

$$
\begin{aligned}
\operatorname{Sparsemax}_{\mathcal V_i}(\boldsymbol r)
&=\underset{\boldsymbol a\in\Delta(\mathcal V_i)}{\arg\min}
\frac12\|\boldsymbol a-\boldsymbol r\|_2^2,\\
\Delta(\mathcal V_i)&=\left\{\boldsymbol a:
a_d\ge0,\ \sum_d a_d=1,\ a_d=0\ (d\notin\mathcal V_i)\right\},\\
\alpha_{i,d}&=M_{i,d}\max(2\bar\alpha_{i,d}-\tau_i,0).
\end{aligned}
\tag{16a}
$$

阈值 $\tau_i$ 由当前分数自适应计算，使权重和为 1，不采用人工设定的 top-k 或固定逐扇区阈值。具体地，将有效分数 $r_d=2\bar\alpha_{i,d}$ 降序排列为 $r_{(j)}$，取满足 $1+jr_{(j)}>\sum_{k=1}^{j}r_{(k)}$ 的最大 $j=K_i$，然后令 $\tau_i=(\sum_{k=1}^{K_i}r_{(k)}-1)/K_i$。无有效文本或背景时单独返回全零权重。

固定缩放系数 2 用于提高相关性分数的差异；若直接对已经位于单纯形上的 $\bar{\boldsymbol\alpha}_i$ 投影，结果会保持原有稠密权重，无法实现排除。系数 2 是当前实现的实验起点，不是经验证的最佳值，不新增可学习参数。所有有效扇区相关性相同时，稀疏投影保持均匀权重，允许全部使用；相关性集中时，低相关扇区的最终权重可严格为零。例如，八扇区相关性为 $[0.45,0.45,0.1/6,\ldots,0.1/6]$ 时，最终权重为 $[0.5,0.5,0,\ldots,0]$。

最终选中集合为 $\mathcal S_i=\{d:\alpha_{i,d}>0\}$，其大小可为 1 至 $|\mathcal V_i|$。选择发生在整句相关性汇总之后：若只对每个 token 单独稀疏化，再平均各 token 的结果，其选中集合的并集仍可能覆盖全部扇区。式（16）保证被最终排除的 Value 对背景汇总没有直接贡献。当前实现仍计算全部扇区特征和相关性，因此这里的选择减少信息混入，不等于跳过采样或节省相同比例的计算。

稀疏投影采用 float32，除支持集变化的边界外是分段可微的，直接使用自动求导，不需要随机采样或直通估计器。只有一个方向被保留时，方向权重局部固定为 1，该投影对分数的局部梯度为零；并不保证训练始终有方向选择梯度。无需构造 $B\times N\times L\times D$ 的中间视觉读取张量。加入稀疏投影后，最终 $e_i$ 不再等价于直接平均逐 token 视觉读取，逐 token 注意力图也不能直接当作实际使用扇区图。

这一层回答“最终使用哪些方向及其相对权重”，不控制背景总用量，也不保证“上方”一词一定准确选择上方扇区。在有效文本与背景均存在时，单纯形投影至少保留一个方向；全部背景都不相关时，仍由下一层候选级门控整体减少背景使用。有效 token 平均仍可能稀释少数关系词的影响，稀疏投影不自动解决该问题。

### 4.4 背景投影与尺度对齐

两路特征幅值差异会影响门控与融合网络的输入。首先将汇总背景投影到候选维度并做无仿射参数的 LayerNorm：

$$
\bar b_i=\operatorname{LN}(W_pe_i),\qquad
W_p:\mathbb R^{128}\rightarrow\mathbb R^{256}.
\tag{17}
$$

使用原候选的均方根幅值作为参考，将背景调整到相近尺度：

$$
r_i=\operatorname{stopgrad}\left(\frac{\|q_i\|_2}{\sqrt C}\right),\qquad
b_i^{ctx}=r_i\bar b_i.
\tag{18}
$$

LayerNorm 的 $\varepsilon=10^{-5}$，当投影特征具有非零方差且方差足够大时，$\bar b_i$ 的均方根接近 1。因此对齐后的背景均方根接近候选均方根；零方差或极小方差时该近似不成立。参考幅值停止梯度，避免尺度对齐引入额外的候选梯度路径。原候选分支保留 $q_i$，不以归一化特征替换。投影 $W_p$ 无偏置，全部背景无效时不会产生常量背景。

### 4.5 背景输入门控与候选—背景非线性融合

定义可用背景指示量 $H_i=H_T\mathbf1[|\mathcal V_i|>0]$。使用候选、背景与句子共同预测背景输入门控：

$$
g_i=H_i\,\sigma\left(
\phi_{mix}([\operatorname{LN}(q_i),\bar b_i,t])\right).
\tag{19}
$$

门控 MLP 的维度为 $640\rightarrow128\rightarrow1$。输入采用归一化候选与背景，减少门控对绝对幅值的依赖。$g_i\in[0,1]$ 仅控制背景进入融合网络的幅值，不规定候选系数为 $1-g_i$。将原候选与门控背景分别放入拼接向量，交由独立的融合 MLP 处理：

$$
q_i^{enh}=\begin{cases}
\phi_{fusion}([q_i,\ g_i b_i^{ctx}]), & H_i=1,\\
q_i, & H_i=0.
\end{cases}
\tag{20}
$$

融合 MLP 为 $512\rightarrow128\rightarrow256$，包含两层带偏置线性映射和中间的 GELU。其输入前半部分始终是未经 $1-g_i$ 缩放的原候选，后半部分是门控后的背景；文本在区域选择和输入门控阶段发挥作用，不直接拼接到该融合 MLP。输出维度与原候选相同，但不执行候选与背景的逐维相加，也不额外叠加候选残差。候选与背景在输出中的影响由融合网络学习，不以两个互补标量表示。

默认门控末层偏置初始化为 $\log(0.1/0.9)$，末层权重以标准差 $10^{-3}$ 的小随机值初始化，使初始背景输入门控接近 0.1，并让门控前层从首步获得梯度。融合 MLP 使用 PyTorch 线性层的默认初始化；门控初值较小并不保证其输出接近原候选，因此本版需重新训练融合网络与 CRS。0.1 是门控初始化值，不是固定融合比例，最终输出也不保证具有与候选相同的范数。

候选分支、背景分支和融合网络共同经过最终指代监督学习。与描述矛盾的背景也可能具有判别价值，因此不规定“不匹配背景必须低门控”。门控不表示候选正确概率或预测贡献比例。对于有效背景，当门控为零时，输出为 $\phi_{fusion}([q_i,0])$：背景输入被移除，候选仍经过学习到的变换，并非自动返回原候选。

当背景全部无效或有效文本为空时，$H_i=0$、$g_i=0$，实现通过显式选择返回原候选表示，即使融合网络具有非零偏置也严格恒等退化。该路径由有效性决定，区别于有效背景下门控恰为零。最终评分为：

$$
s_i=\operatorname{CRS}(q_i^{enh},T,m^{text}).
\tag{21}
$$

为区分融合权重与实际特征改变量，记录：

$$
R_i=
\frac{\|q_i^{enh}-q_i\|_2}
{\max(\|q_i\|_2,10^{-6})}.
\tag{22}
$$

该值描述候选表示改变多少，不能直接解释为准确率提升或背景贡献率；其定义与旧版背景残差范数的含义也不同。

### 4.6 合理性与适用范围

八扇区分别保留区域视觉内容，原始文本注意力估计相关性，最终稀疏投影允许直接排除部分扇区。稀疏区域权重控制背景内部的选择，输入门控控制汇总背景的幅值，拼接 MLP 学习候选与背景的非线性组合。单纯形投影本身不能整体关闭有效背景。尺度对齐稳定两路输入，较小门控初值减少初始背景幅值，无效背景保持恒等输出；这些措施并不使有效背景下的新融合网络成为恒等映射。结构只使用小型投影、共享视觉 MLP、一次单头文本到扇区读取、八维稀疏投影及两个小型门控／融合 MLP，不引入额外方向提示、序列编码器或额外监督。

该结构仍有需要实验检验的限制：八个方向最终压缩为一个向量，无法保证完整保存多个关系；拼接输入保留候选，并不保证后续非线性映射完整保留其身份信息；128 维中间层会进一步压缩两路内容。需要观察以外观为主的表达性能，并对比旧线性融合及容量相当的候选单路 MLP。这里给出的是可实现的机制假设，不预先认定拼接 MLP 能提升准确率。

## 5 训练目标与完整数据流

训练和推理都使用同一候选集合下的预测框与候选表示，采样几何使用：

$$
\hat b_i=\operatorname{stopgrad}(b_i).
\tag{23}
$$

候选表示、视觉特征、文本特征及 BCEM 参数正常接收梯度。检测回归继续由原检测监督控制；BCEM 不直接修改本次前向的候选框、检测类别或筛选索引。联合训练仍可能通过共享骨干间接影响后续候选分布。

沿用现有 Hungarian 匹配与指代标签，指代损失为：

$$
\mathcal L_{ref}
=-\frac1N\sum_i
\left[y_i\log s_i+(1-y_i)\log(1-s_i)\right].
\tag{24}
$$

总体目标保持既有设置：

$$
\mathcal L=
\lambda_{det}\mathcal L_{det}
+\lambda_{ref}\mathcal L_{ref}
+\mathcal L_{\mathrm{existing\ branches}}.
\tag{25}
$$

不新增背景类别、环境分割、词语对应或背景排序监督。多个候选可能处于同样符合描述的环境，因此背景不承担独立确定正确目标的任务。

~~~mermaid
flowchart TD
    I["图像与完整表达"] --> ENC["BEiT-3 图文编码与融合"]
    ENC --> FPN["FPN 多尺度视觉特征"]
    ENC --> TEXT["上下文化文本特征"]
    FPN --> DEC["候选生成与可选 Grab 筛选"]
    DEC --> BOX["对应预测框：采样时停止梯度"]
    DEC --> ADP["refer_adapter"]
    ADP --> Q["原始候选表示 q"]
    FPN --> FM["最高分辨率检测特征：投影到 128 维"]
    BOX --> SECTOR["8 个框外扇区：每条射线延伸到有效图像边缘"]
    FM --> SECTOR
    MASK["有效图像尺寸与视觉 mask"] --> SECTOR
    SECTOR --> POOL["每扇区 5×5 点近似面积池化"]
    POOL --> BG["共享视觉 MLP＋LayerNorm：不附加扇区位置"]
    TEXT --> TOK["原始表达的逐 token 投影"]
    TOK --> SENT["有效 token 平均：供最终融合"]
    TM["文本 padding mask"] --> SENT
    TOK --> READ["文本 Query 读取扇区视觉 Key／Value"]
    TM --> READ
    BG --> READ
    SECTOR --> VM["无效扇区屏蔽"]
    VM --> READ
    READ --> REL["有效 token 注意力平均：八方向相关性"]
    REL --> SELECT["Sparsemax：选择数量自适应，未选方向权重为 0"]
    SELECT --> SUM["仅汇总被选扇区的 Value：背景 e"]
    BG --> SUM
    SUM --> ALIGN["无偏置投影＋LayerNorm＋候选 RMS 尺度对齐"]
    ALIGN --> GATE["候选／背景／文本 MLP：背景输入门控 g"]
    Q --> GATE
    SENT --> GATE
    GATE --> FUSE["候选与 g×背景拼接：512→128→256 MLP"]
    ALIGN --> FUSE
    Q --> FUSE
    FUSE --> CRS["CRS 最终指代评分"]
    TEXT --> CRS
    TM --> CRS
~~~

当前训练与推理均在 REFHead 的 refer_adapter 之后、mrsm 之前接入。若启用 Grab，使用筛选后对应的查询和框；图像上下文仍从最高分辨率检测特征读取。GTMHead 未在本次实现中接入。

## 6 实现接口、配置与运行

### 6.1 张量与诊断信息

设批量大小为 $B$、候选数为 $N$、文本长度为 $L$：

| 数据 | 形状 |
| --- | --- |
| 原候选表示 | $B\times N\times256$ |
| 归一化预测框 | $B\times N\times4$ |
| 文本特征 | $B\times L\times256$ |
| 采样点 | $B\times N\times8\times25\times2$ |
| 扇区背景向量 | $B\times N\times8\times128$ |
| 投影文本 token | $B\times L\times128$ |
| 文本到扇区注意力 | $B\times N\times L\times8$ |
| 方向重要性 | $B\times N\times8$ |
| 汇总背景 | $B\times N\times128$ |
| 对齐后的背景 | $B\times N\times256$ |
| 背景输入门控 | $B\times N\times1$ |
| 候选与门控背景拼接 | $B\times N\times512$ |
| 增强候选表示 | $B\times N\times256$ |

实现文件：

- [BCEM 模块](propvg/models/heads/background_context.py)
- [REFHead 接入](propvg/models/heads/REFHead.py)
- [现有 AerialVG 配置](configs/aerialvg/PropVG-aerialvg.py)
- [CPU 回归测试](tools/test_bcem.py)

模块返回增强候选表示和停止梯度的诊断信息。extra_dict 中包含：

| 字段 | 含义 |
| --- | --- |
| bcem_context | 汇总背景表示 |
| bcem_direction_weights | 最终八方向稀疏权重；未选中方向严格为零，有效样本权重和为 1 |
| bcem_direction_weights_dense | 稀疏选择之前的有效 token 平均注意力 |
| bcem_selected_sectors | 实际使用的扇区布尔掩码，形状为 $B\times N\times8$ |
| bcem_selected_sector_count | 每个候选实际使用的方向数，形状为 $B\times N$ |
| bcem_text_sector_attention | 稀疏选择之前的逐 token 注意力，形状为 $B\times N\times L\times8$；padding 或全部无效背景行均为零 |
| bcem_background_weight | 候选级背景输入门控 $g_i$，无有效背景或文本时为零；不表示输出贡献比例 |
| bcem_valid_sectors | 八个扇区的有效性 |
| bcem_sector_coverage | 每个扇区的有效采样点比例，不代表精确面积覆盖率 |
| bcem_residual_ratio | 实际候选改变量与原候选范数之比，即式（22） |

### 6.2 默认配置与开销

| 参数 | 默认值 |
| --- | --- |
| context_dim | 128 |
| angular_samples、radial_samples | 5、5 |
| min_valid_points | 5 |
| fusion_init | 0.1：可学习门控的初始背景输入强度 |
| chunk_size | 32 |
| use_position | False：默认不附加方向或相对位置编码 |
| use_text_attention | True |
| use_sparse_selection | True：整句相关性汇总后进行稀疏选择 |
| use_learnable_fusion | True：学习背景输入门控；False 时固定该门控，拼接融合 MLP 仍可学习 |

模块与配置默认关闭额外位置编码。当前 AerialVG 实验配置的 enable_bcem=False，运行新版背景融合时须显式启用 BCEM。REFHead 构造默认关闭 BCEM，以保留其他原任务的行为；关闭时不分配背景分支参数。sector_radius 已从模块构造及当前配置移除，覆盖长度完全由候选框与有效图像边界决定。

默认拼接 MLP 融合版包含 345,217 个可训练参数；新增融合 MLP 为 98,688 个参数，稀疏投影不增加参数。关闭稀疏选择的稠密对照具有相同参数及初始化。可选位置编码对照为 378,753 个，默认无位置编码条件下的均匀扇区读取为 328,833 个，固定背景输入门控为 263,040 个。均匀读取与固定门控同时启用时为 213,888 个。各个版本均保留拼接融合 MLP，不再分配文本乘法调制、额外方向评分 MLP 或候选方向投影。

每个候选读取 $8\times25=200$ 个点，仅保留八个区域特征。整幅视觉特征只投影一次，然后按候选分块读取。扇区池化开销约为 $O(BN\cdot8AKD)$，逐 token 投影约为 $O(BLCD)$，共享视觉映射与 Key、Value 投影约为 $O(BN\cdot8D^2)$，文本注意力约为 $O(BN\cdot8LD)$，输入门控约为 $O(BN(2C+D)D)$，拼接融合 MLP 约为 $O(BN\cdot3CD)$。仅构造文本与八个扇区之间的注意力，不对扇区内部每个采样点分别计算文本注意力，也不构造逐 token 视觉读取的大型中间张量。当前分支不复制完整特征图到每个候选，不使用多层交互或激活重计算。

参数与结构减少不等同于已经测得的延迟或显存收益，实际 GPU 开销仍需测量。

稀疏选择额外处理每候选八个分数，排序复杂度为 $O(BN\cdot8\log8)$。其目的为排除不相关扇区的 Value，并不减少当前实现中八扇区的采样和 Key、Value 计算。

### 6.3 运行与消融

在已安装项目依赖的环境中，从仓库根目录执行：

~~~bash
# 现有配置直接训练文本稀疏选择扇区 BCEM
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py --launcher none --cfg-options model.head.enable_bcem=True

# PropVG 基线
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_bcem=False

# 可选对照：额外加入显式方向位置编码
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_bcem=True model.head.bcem.use_position=True

# 稠密注意力对照：恢复全部有效方向的软权重
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_bcem=True model.head.bcem.use_sparse_selection=False

# 使用有效方向的均匀权重，保留可学习融合
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_bcem=True model.head.bcem.use_text_attention=False

# 固定背景输入门控为 0.1，拼接 MLP 仍学习融合
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_bcem=True model.head.bcem.use_learnable_fusion=False model.head.bcem.fusion_init=0.1

# 同时使用均匀背景读取和固定背景输入门控
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_bcem=True model.head.bcem.use_text_attention=False model.head.bcem.use_learnable_fusion=False

# 从同结构 AerialVG 基线初始化
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --finetune-from /path/to/aerialvg-baseline.pth --cfg-options model.head.enable_bcem=True

# 使用当前文本读取视觉版训练后的权重推理
PYTHONPATH=. python tools/test.py configs/aerialvg/PropVG-aerialvg.py \
  --load-from /path/to/text-attention-bcem-checkpoint.pth --launcher none --cfg-options model.head.enable_bcem=True

# CPU 回归检查
PYTHONPATH=. python tools/test_bcem.py
~~~

默认 use_position=False，八扇区采样保持不变，不分配位置 MLP，扇区映射仅接收视觉。use_position=True 仅供历史位置方案对照，额外编码主方向的正弦／余弦与有效区域中心相对候选的偏移，并与视觉拼接；这些数值不注入原始文本。use_sparse_selection=False 时仅取消稀疏投影，恢复有效 token 平均的稠密方向权重，保留相同参数及其初始化。use_text_attention=False 时移除 Key 投影，对有效扇区采用均匀读取，同时绕过稀疏投影；句子特征仍参与可学习输入门控，使该对照只取消基于文本内容的扇区选择。为沿用现有配置接口，use_learnable_fusion 仍保留，但在本版仅控制背景输入门控是否学习：False 时移除 fusion_gate，将输入门控固定为 fusion_init，feature_fusion 拼接 MLP 始终保留并参与学习。无效背景仍严格退化为候选分支。当文本注意力与可学习门控同时关闭时无需文本投影，背景分支均匀汇总，采用固定输入门控和可学习拼接融合；基线中的 BEiT-3 与 CRS 图文交互继续保留。

原训练脚本继续使用现有配置。旧版 context_strength 已被 fusion_init 及可学习门控替换；两者含义不同，不可将旧系数直接解释为新的背景输入门控。旧的 use_text_modulation、use_direction_weighting 配置项不再使用，以 use_text_attention 控制内容相关读取。当前使用 sector_key 和 sector_value，并删除 sector_query 与 modulation；注意力归一化轴也由文本改为扇区。因此原转移版、固定残差版、四路拼接版及文本乘法版 BCEM checkpoint 不能直接作为同结构 resume 或同结构推理；恢复优化器只适用于当前版本 checkpoint。从不含 BCEM 的同结构 AerialVG 基线启动时采用 finetune-from。若复用旧 BCEM 实验的骨干与候选生成权重，应先筛除全部 head.bcem 权重，并重新初始化当前分支；strict=False 本身不会跳过同名但形状不同的张量。本次未修改旧实验保存的配置与代码副本。

## 7 实验验证与适用边界

本版新增 feature_fusion 参数；历史加权相加版没有该网络，不能作为完整同结构 checkpoint 严格恢复模型和优化器。即使旧参数形状一致，部分加载后新融合网络仍是随机初始化，必须继续训练它和 CRS，不能把该结果当作原模型等价推理。建议从同设置的无 BCEM 基线初始化，或复用旧实验的骨干与候选分支并重新初始化整个 BCEM。历史 use_position=True 的版本还具有不同的 sector_fusion 首层形状，strict=False 不能跳过同名张量的形状差异；需要筛除旧 BCEM 权重。效果对照应独立训练，不将推理时临时更换融合方式的分数变化当作训练收益。

本次从固定长度改为图像边界覆盖仅改变几何与池化权重，不新增可训练参数；上一版无位置编码、拼接 MLP 的模型参数形状仍兼容。但是读取范围发生变化，需要重新训练或微调，不能把临时换采样范围的推理结果当作独立训练效果。旧保存配置含有 sector_radius 时不能直接用于当前构造，应采用当前配置；旧实验配置与源码副本保持原样。

核心实验收敛为以下对照：

| 版本 | 要回答的问题 |
| --- | --- |
| PropVG | 原始基线性能 |
| 八扇区视觉背景，均匀区域权重＋固定输入门控＋拼接 MLP | 增加候选外部背景是否有效 |
| 八扇区视觉背景，文本选择＋固定输入门控＋拼接 MLP | 原始文本依据内容选择扇区是否有价值 |
| 八扇区视觉背景，均匀区域权重＋学习输入门控＋拼接 MLP | 候选级背景输入门控是否有价值 |
| 稠密文本注意力＋学习输入门控＋拼接 MLP | 全部有效方向参与的软读取效果 |
| 完整 BCEM：稀疏文本选择＋学习输入门控＋拼接 MLP | 排除部分扇区是否优于稠密读取 |
| 相同采样与文本选择，旧加权相加与当前拼接 MLP 分别训练 | 非线性融合是否优于旧线性混合 |
| 相同容量的候选单路 MLP，背景输入固定为零 | 收益是否只是来自新增网络容量 |
| 完整 BCEM，额外启用位置编码 | 附加方向与相对位置编码是否有价值 |

上述版本必须分别训练，让各自的 CRS 适应对应候选表示。不能只在带背景模型推理时关闭分支，将其掉点直接作为相对于独立 PropVG 基线的提升；该操作会改变 CRS 的输入分布，只能作为分支依赖性诊断。背景错配扰动也只能作为诊断，不能替代独立训练对照。

不同消融会改变参数数量，应报告真实参数与开销，尤其应增加容量相当的候选单路 MLP 对照。该容量对照和旧线性融合属于待开展实验，不是当前配置开关已实现的分支。统一候选数、数据划分、教师文件、教师去重与 NMS 阈值、教师框数、初始化、优化器、学习率日程、随机种子、训练预算与覆盖范围。固定背景输入门控对照默认采用 0.1，使其与可学习门控的初始化一致；其他固定值只在验证集上选择。建议报告多个随机种子的均值与波动。

还应开展固定候选实验：从同一基线固定预测框、候选表示及对应图文特征，冻结候选生成与骨干；分别重新训练各版本的 CRS 与其启用的背景分支，保持初始化及训练预算一致。分别报告候选覆盖率、已有候选覆盖 GT 时的条件选择准确率，以及最终 Top-1、Top-5 和框 mIoU。这能区分背景对评分的帮助与联合训练对候选生成的间接影响，并避免 CRS 对单一特征分布的适应造成误判。

样本分析重点包括场景环境关系、辅助对象关系、多个关系以及图像边缘背景缺失。分别可视化稀疏选择之前的 token 注意力与最终实际使用的扇区，统计最终权重的分布与熵、选中方向数及其相对于有效方向数的比例、候选级背景输入门控、有效点比例及候选改变量；特别检查是否能够按表达选择一两个方向或全部方向，以及外观为主的表达是否出现门控过大和性能下降。注意力与权重只能作为诊断线索，不能直接当作正确解释。当前训练、验证、测试分别指向 vg_train_odvg.jsonl、vg_val_odvg.jsonl、vg_test_odvg.jsonl；超参数与模型选择使用验证集，测试集用于最终报告。

方法保留了八方向区域组织，但扇区内部经过池化，因此不以恢复环境转移或精确距离顺序为贡献。稀疏 5×5 读取可能遗漏细小辅助对象；最高分辨率 FPN 也不能恢复骨干已经丢失的细节。文本到扇区注意力不保证严格绑定多个关系条件，也不保证包含 top 的表达一定选择上方扇区；全部有效 token 的平均可能稀释关键关系词。背景相似的候选仍需原候选表示与 CRS 完成区分。

实现验证覆盖框外扇区几何、面积池化、实际区域中心、padding 不变性、逐 token 的有效扇区归一化、默认选择不依赖额外位置数值及其扇区排列等变性、可选位置分支对方向选择的影响、文本 Query 对视觉 Key 的点积对齐及屏蔽、单有效扇区时只返回视觉 Value、均匀读取的隔离效果、背景尺度对齐、输入门控初始化、拼接前候选输入不随门控缩小、零门控移除背景输入依赖、无效背景即使有融合偏置也保持恒等、Query／Key／Value 与门控和拼接网络梯度传播、候选分块和 REFHead 训练／推理接入。CPU 测试说明计算行为符合设计，不能替代完整数据集效果及 GPU 开销验证。

本次新增检查覆盖选择全部／一个／两个／三个／四个方向、无效区域与空行、稀疏投影自动求导的有限差分校验、未选中扇区的 Value 不影响最终背景，以及实际使用数量的诊断一致性。CPU 检查同时覆盖 REFHead 训练、反向及推理接入。此前稠密文本注意力版还完成了 586×586 padding 画布、256 个候选和 64 个文本 token 的合成训练与推理；该历史检查不能代替新选择机制的训练效果评估。本次未执行完整数据集训练或性能评估。

本次几何检查新增每条子射线在非正方形图像上到达图像边缘、不同射线长度对应的面积权重，以及候选框覆盖整图时无背景恒等退化。当前 34 项 BCEM、6 项可选流水线及 5 项 CRS 文本池化检查共 45 项全部通过。本次未修改 README、未创建独立配置文件，文档仍位于仓库根目录。

## 参考资料

1. Dai M., et al. PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination. ICCV, 2025. [论文](https://arxiv.org/html/2509.04833v1)，[官方代码](https://github.com/Dmmm1997/PropVG)。
2. Liu J., et al. AerialVG: A Challenging Benchmark for Aerial Visual Grounding by Exploring Positional Relations. arXiv:2504.07836. [论文](https://arxiv.org/html/2504.07836v3)。
