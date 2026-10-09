# 面向航拍视觉指代的候选背景上下文增强方法

**模块名称：** 背景上下文增强模块（Background Context Enhancement Module，BCEM）。

**研究定位：** 面向航拍场景中外观相似目标的指代消歧，研究如何将候选框外的场景环境及辅助对象信息组织为可供指代评分使用的补充证据。本方法在 PropVG 的候选表示与 Contrastive-based Refer Scoring（CRS）之间增加背景增强分支，不新增环境类别、分割或词语对应标注。

**本文状态：** 本文描述已实现的方法及其验证设计。采样尺度固定为有效图像短边的 10%、15%、20%、25%；文中不将待验证的机制假设表述为已取得的性能结论。

## 1 研究背景与问题提出

### 1.1 航拍视觉指代中的目标证据不足

航拍图像通常具有视野广、目标密集和目标尺寸较小等特点。同类车辆或行人在颜色、形状及局部纹理方面容易表现出高度相似性，仅依赖目标自身的视觉证据，难以从多个候选中确定自然语言描述所指向的对象。与此相对应，指代表达往往通过道路、停车场、建筑等场景元素，以及周围车辆、行人等辅助对象，进一步限定目标的位置与身份。

例如，对于“停车场内的白色轿车，上方有一辆白色 SUV”这一表达，候选车辆的颜色和类别只能提供初步匹配依据，停车环境与辅助车辆的相对位置共同构成了消歧证据。AerialVG 将相似目标之间的空间关系作为任务的重要挑战，其论文也指出，道路等非结构化背景信息仍难以被现有模型充分利用。[AerialVG 原始论文](https://arxiv.org/html/2504.07836v3)

因此，研究对象不应局限于目标外观，还应包含候选周围可观测的场景与辅助对象信息，以及这些信息相对于候选的空间组织。

### 1.2 现有隐式上下文交互的表示缺口

PropVG 使用 BEiT-3 完成视觉语言编码与融合，并通过多尺度可变形解码器生成候选。因此，其候选表示已经能够隐式利用图像上下文，不能将研究动机表述为“PropVG 没有背景信息”。[PropVG 原始论文](https://arxiv.org/html/2509.04833v1)

本研究关注的表示缺口是：现有候选表示没有独立、显式地组织每个候选框外的背景内容、相对位置和邻近尺度之间的变化。将较大区域整体池化为单个向量，可能弱化不同环境元素相对于候选的空间分布；在背景内容相似但位置组织不同的情况下，这种弱化会影响指代消歧。

据此，BCEM 将背景信息作为候选相关的补充证据进行建模。方向与尺度用于确定采样布局，相邻变化用于建立背景片段之间的结构联系，完整文本用于条件化证据读取。模块最终完成候选表示增强，而不要求显式输出方向类别、背景类别或词语对应关系。

### 1.3 研究目标与机制假设

本方法围绕以下三个问题展开：

1. **背景获取：** 如何在较大范围内读取候选框外的上下文，同时保留其相对于候选的空间位置？
2. **结构组织：** 在背景特征与准确坐标之外，同方向相邻尺度的显式连接及顺序累积是否有助于消歧？
3. **自适应使用：** 如何依据完整表达和当前候选，控制背景证据对指代评分的影响程度？

对应的机制假设分别为：候选相关的多尺度背景读取能够补充目标线索；相邻连接与顺序编码能够提供适合空间环境判别的归纳偏置；自适应使用权重能够减少无关背景对候选表示的干扰。这些假设需要通过递进消融与固定候选实验验证。

## 2 总体技术路线与符号定义

设有效图像为 $I\in\mathbb R^{H\times W\times3}$，补边后的输入画布尺寸为 $\widetilde H\times\widetilde W$。有效图像短边记为：

$$
S=\min(H,W).
\tag{1}
$$

候选 $i$ 的预测框为 $b_i=(c_i^x,c_i^y,w_i,h_i)$，此处四个量已经转换为有效图像坐标下的像素值。其经 refer_adapter 投影后的表示为 $q_i\in\mathbb R^C$。完整表达对应的上下文化文本特征为：

$$
T=[t_1,\ldots,t_L],\qquad t_\ell\in\mathbb R^C.
\tag{2}
$$

最高分辨率检测侧 FPN 特征记为 $F\in\mathbb R^{C\times H_f\times W_f}$。首版采用 $C=256$、背景表示维度 $D=128$。

BCEM 的总体映射为：

$$
(F,b_i,q_i,T,\mathcal M)
\longmapsto(e_i,g_i)
\longmapsto q_i^{\mathrm{enh}}
\longmapsto s_i,
\tag{3}
$$

其中 $\mathcal M$ 包含图像与文本有效性信息，$e_i$ 为背景上下文表示，$g_i$ 为背景使用权重，$s_i$ 为 CRS 产生的最终指代分数。

需要说明的是，$F$ 和 $T$ 均来自已有图文融合过程；“背景”表示采样位置位于候选框外，而不表示特征已经与目标内容完全解耦。

## 3 候选相关的四尺度背景读取

### 3.1 方向布局与尺度定义

设置八个主方向：

$$
\Theta=
\left\{-\frac{\pi}{2},\frac{\pi}{2},\pi,0,
-\frac{3\pi}{4},-\frac{\pi}{4},\frac{3\pi}{4},\frac{\pi}{4}\right\}.
\tag{4}
$$

图像横轴向右、纵轴向下，因此上述顺序依次对应上、下、左、右、左上、右上、左下、右下。四个径向尺度为：

$$
\boldsymbol\alpha=(0.10,0.15,0.20,0.25),\qquad
s_r=\alpha_rS,\quad r=1,\ldots,4.
\tag{5}
$$

这里 $s_r$ 表示沿射线从候选框边界向外移动的距离，既不是从候选中心计算的半径，也不是采样窗口的边长。以有效图像短边为尺度基准，可以使背景覆盖范围不依赖小目标框的宽高。

| 有效图像短边 | 10% | 15% | 20% | 25% |
| --- | --- | --- | --- | --- |
| 1000 像素 | 100 | 150 | 200 | 250 |
| 330 像素 | 33 | 49.5 | 66 | 82.5 |

四个尺度主要面向较大范围的场景与辅助对象信息，不直接承担精细路缘或极近邻边界的完整重建任务。

### 3.2 框边界交点与局部采样区域

为减少单条细射线的漏采，每个主方向使用三个子角度：

$$
\delta_a\in\{-15^\circ,0,15^\circ\},\qquad
\boldsymbol d_{d,a}
=[\cos(\theta_d+\delta_a),\sin(\theta_d+\delta_a)].
\tag{6}
$$

令候选中心为 $\boldsymbol c_i$，半宽高为 $(w_i/2,h_i/2)$，子射线与候选框边界的交点距离为：

$$
\lambda_{i,d,a}
=\min\left\{
\frac{w_i/2}{\max(|d^x_{d,a}|,\varepsilon)},
\frac{h_i/2}{\max(|d^y_{d,a}|,\varepsilon)}
\right\}.
\tag{7}
$$

其中数值稳定常数取 $\varepsilon=10^{-7}$。每个尺度再取三个径向位置。定义局部径向半宽：

$$
\beta_r=\frac14\min
\{\alpha_r-\alpha_{r-1},\alpha_{r+1}-\alpha_r\},
\tag{8}
$$

其中 $\alpha_0=0$，末端使用
$\alpha_5=\alpha_4+(\alpha_4-\alpha_3)$。
在当前四尺度设置下，各尺度的 $\beta_r$ 均约为 0.0125。

九个采样点为：

$$
\boldsymbol p_{i,d,r,a,k}
=\boldsymbol c_i+
\left[\lambda_{i,d,a}
+(\alpha_r+k\beta_r)S\right]\boldsymbol d_{d,a},
\quad k\in\{-1,0,1\}.
\tag{9}
$$

每个子射线均从自己的框边界交点向外采样，适用于长宽不同的候选框。由于局部偏移小于中心框外距离，几何采样点保持在候选框外。

三角度与三径向位置共同构成一个稀疏小区域，并非对整个扇区的密集覆盖。局部区域内部汇总为一个向量，因此证据数量仍为 $8\times4=32$。

### 3.3 有效性约束与局部视觉汇总

采样位置首先转换为 padding 画布对应的归一化网格坐标：

$$
\boldsymbol g(\boldsymbol p)=
\left[
2p^x/\widetilde W-1,\
2p^y/\widetilde H-1
\right].
\tag{10}
$$

实现使用双线性插值及 align_corners=False。有效图像尺寸用于计算背景尺度与几何越界；padding 尺寸用于将像素坐标映射至完整特征画布。二者必须分别处理。

在零化视觉 padding 后，通过共享 $1\times1$ 投影得到 $\widetilde F=W_fF$，将特征维度从 256 降至 128。每个采样点的有效性由以下条件共同决定：

- 位于有效图像内部；
- 位于候选框之外，且候选框宽高均为正；
- 双线性插值的支撑区域不接触视觉 mask 标记的 padding 或特征网格外部。实现通过插值有效性图得到支撑权重 $\eta$，要求 $\eta\ge1-10^{-6}$。

记点有效性为 $m_{i,d,r,j}\in\{0,1\}$，有效点数为：

$$
n_{i,d,r}=\sum_{j=1}^{9}m_{i,d,r,j}.
\tag{11}
$$

一个背景片段至少具有五个有效点才保留：

$$
M^{node}_{i,d,r}=\mathbf1[n_{i,d,r}\ge5].
\tag{12}
$$

局部背景视觉表示为：

$$
u_{i,d,r}
=M^{node}_{i,d,r}\operatorname{LN}
\left(
\frac{
\sum_{j=1}^{9}m_{i,d,r,j}
\widetilde F(\boldsymbol p_{i,d,r,j})
}{
\max(1,n_{i,d,r})
}
\right).
\tag{13}
$$

几何屏蔽能够约束读取位置，但不消除骨干感受野及插值邻域中的目标信息。因此，不采用“扩展区域特征减去目标特征”来宣称提取纯背景。

## 4 背景位置与相邻变化的结构化表示

### 4.1 空间位置编码

取零角度偏移、零径向偏移对应的位置为片段名义中心 $\bar{\boldsymbol p}_{i,d,r}$。令 $(\Delta x,\Delta y)=\bar{\boldsymbol p}_{i,d,r}-\boldsymbol c_i$，该位置到候选框的最短距离为：

$$
\rho_{i,d,r}
=\sqrt{
\max(|\Delta x|-w_i/2,0)^2+
\max(|\Delta y|-h_i/2,0)^2
}.
\tag{14}
$$

沿射线的框外距离与最短框边界距离通常不同，二者均保留。位置数值向量定义为：

$$
z_{i,d,r}=
\left[
\sin\theta_d,\cos\theta_d,\alpha_r,\frac{\rho_{i,d,r}}S,
\frac{\Delta x}S,\frac{\Delta y}S,
\frac{\bar p^x_{i,d,r}}W,\frac{\bar p^y_{i,d,r}}H,
\frac{w_i}W,\frac{h_i}H,\frac{n_{i,d,r}}9
\right]\in\mathbb R^{11}.
\tag{15}
$$

原始背景证据为：

$$
a_{i,d,r}
=u_{i,d,r}+\phi_{pos}(z_{i,d,r})+\tau_{node},
\tag{16}
$$

其中 $\phi_{pos}$ 是两层 MLP，$\tau_{node}$ 是可学习的证据类型向量。

### 4.2 相邻背景变化编码

同方向的四个背景片段形成三条相邻连接，即
10%→15%、15%→20%、20%→25%。首先编码连接的几何属性：

$$
p_{i,d,r}
=\phi_{pair}
\left[
z_{i,d,r},z_{i,d,r+1},\alpha_{r+1}-\alpha_r
\right]\in\mathbb R^D.
\tag{17}
$$

然后联合两个端点及其差分形成转移表示：

$$
v_{i,d,r}
=\phi_{tr}
\left[
u_{i,d,r},u_{i,d,r+1},
u_{i,d,r+1}-u_{i,d,r},p_{i,d,r}
\right].
\tag{18}
$$

两个原始端点使网络能够判断变化发生在何种特征之间，差分为变化建模提供便捷的参数化形式。这里的差分不承担背景分离作用，也没有引入端点之外的新观测信息；在拼接后的首层线性映射中，差分项还可以被端点权重吸收。因此，潜在贡献应归因于相邻连接和后续顺序累积的归纳偏置，而非差分本身提供了新的语义信息。

### 4.3 顺序累积与缺失观测处理

相邻连接的有效性为：

$$
M^{tr}_{i,d,r}
=M^{node}_{i,d,r}M^{node}_{i,d,r+1}.
\tag{19}
$$

每个候选、每个方向分别初始化零状态，使用共享 GRUCell 按由近到远的顺序编码：

$$
h_{i,d,0}=0,\qquad
h_{i,d,r}
=M^{tr}_{i,d,r}
\operatorname{GRUCell}(v_{i,d,r},h_{i,d,r-1}).
\tag{20}
$$

当连接无效时，状态归零，避免将缺失观测两侧错误地视为连续路径。八个方向共享网络参数，但不共享运行时状态。

保留所有中间输出，并构造：

$$
b^{tr}_{i,d,r}
=h_{i,d,r}+p_{i,d,r}+\tau_{tr}.
\tag{21}
$$

由此，完整背景证据集合为：

$$
E_i=
\{a_{i,d,r}\}_{d=1,r=1}^{8,4}
\cup
\{b^{tr}_{i,d,r}\}_{d=1,r=1}^{8,3},
\qquad |E_i|=32+24=56.
\tag{22}
$$

式（22）描述的是稀疏背景观测及其连接结构，不表示恢复了连续环境或获得了道路、建筑的类别标注。

## 5 文本条件下的背景读取与候选增强

### 5.1 带空证据的背景读取

完整文本 token 经共享投影形成 Query，背景证据分别形成 Key 和 Value。采用四头注意力，每头维度 $D_h=32$。

对第 $h$ 个头，可写为：

$$
Q_\ell^h=W_Q^ht_\ell,\quad
K_{i,j}^h=W_K^hE_{i,j},\quad
V_{i,j}^h=W_V^hE_{i,j}.
\tag{23}
$$

追加空证据 $j=0$，其投影空间 Key 可学习、Value 显式固定为零：

$$
K_{i,0}^h=k_0^h,\qquad V_{i,0}^h=0.
\tag{24}
$$

令 $\mathcal V_i$ 为有效背景证据的索引集合，注意力权重为：

$$
A_{i,\ell,j}^h
=\frac{
\exp\big((Q_\ell^h)^\top K_{i,j}^h/\sqrt{D_h}\big)
}{
\sum_{k\in\mathcal V_i\cup\{0\}}
\exp\big((Q_\ell^h)^\top K_{i,k}^h/\sqrt{D_h}\big)
},
\quad j\in\mathcal V_i\cup\{0\}.
\tag{25}
$$

读取结果为：

$$
o_{i,\ell}
=W_O\operatorname{Concat}_{h=1}^{4}
\left[
\sum_{j\in\mathcal V_i\cup\{0\}}
A_{i,\ell,j}^hV_{i,j}^h
\right].
\tag{26}
$$

无效背景证据不进入式（25）的分母。实现将零 Value 追加在投影之后，输出投影不使用偏置，也不添加文本残差，因此全部真实背景证据无效时，读取结果严格为零。

令 $m_\ell^{text}=1$ 表示有效文本位置，背景上下文表示为：

$$
e_i=
\frac{
\sum_{\ell=1}^{L}m_\ell^{text}o_{i,\ell}
}{
\max(1,\sum_{\ell=1}^{L}m_\ell^{text})
}\in\mathbb R^D.
\tag{27}
$$

该机制允许表达中的不同 token 软读取背景信息，但不保证特定词语被准确路由到预期片段。空证据允许模型减少背景读取，同样需要通过训练获得合适的使用行为。

### 5.2 背景使用权重与残差融合

整句文本表示采用有效位置平均：

$$
\bar T=
\frac{\sum_{\ell=1}^{L}m_\ell^{text}t_\ell}
{\max(1,\sum_{\ell=1}^{L}m_\ell^{text})}.
\tag{28}
$$

候选相关的背景使用权重定义为：

$$
g_i=\sigma\big(\phi_g([q_i,e_i,\bar T])\big)\in[0,1].
\tag{29}
$$

其中门控 MLP 的维度为 $640\rightarrow128\rightarrow1$。环境投影 $W_P:\mathbb R^{128}\rightarrow\mathbb R^{256}$ 不使用偏置，最终融合为：

$$
q_i^{enh}=q_i+\gamma g_iW_Pe_i,\qquad \gamma=0.1.
\tag{30}
$$

首版将 $\gamma$ 固定，以减少整体融合强度和门控同时变化造成的解释困难。背景权重表示信息使用程度，不是候选正确概率。与描述矛盾的环境也可能具有排除候选的价值，因此不规定不匹配环境必须对应较低的 $g_i$。

最终评分仍由原 CRS 完成：

$$
s_i=\operatorname{CRS}(q_i^{enh},T,m^{text}).
\tag{31}
$$

当背景无效时，$e_i=0$，因此式（30）严格退化为原候选表示。需要监测真实残差比例：

$$
R_i=
\frac{\|\gamma g_iW_Pe_i\|_2}
{\max(\|q_i\|_2,10^{-6})}.
\tag{32}
$$

固定 $\gamma=0.1$ 不等同于背景的实际贡献恒为候选范数的 10%；式（32）用于反映训练后的真实尺度。

## 6 训练目标与 PropVG 接入

### 6.1 保留原有监督

训练与推理均使用最后一层预测框。构造采样几何时对预测框停止梯度，避免指代损失直接通过移动采样位置干扰框的几何含义。视觉特征、文本特征、原候选表示以及 BCEM 参数继续接收梯度：

$$
\hat b_i=\operatorname{stopgrad}(b_i).
\tag{33}
$$

保留现有 Hungarian 匹配与指代标签。令 $y_i\in\{0,1\}$ 表示候选是否被匹配为所指目标，当前指代损失为：

$$
\mathcal L_{ref}
=-\frac1N\sum_{i=1}^{N}
\big[y_i\log s_i+(1-y_i)\log(1-s_i)\big].
\tag{34}
$$

总体目标沿用既有设置：

$$
\mathcal L=
\lambda_{det}\mathcal L_{det}
+\lambda_{ref}\mathcal L_{ref}
+\mathcal L_{\mathrm{existing\ branches}}.
\tag{35}
$$

AerialVG 当前以检测和指代监督为主；已有筛选或分割分支是否启用由原配置控制。BCEM 不新增背景分类损失、环境分割损失，也不构造“GT 背景必须优于所有非 GT 背景”的约束，因为多个候选可能具有同样符合描述的环境。

### 6.2 训练与推理的共同接入位置

当前实现针对 REFHead，在训练与推理中均执行：

1. Decoder 产生候选查询和预测框。
2. 若启用已有 Grab 筛选与绑定分支，同步选取候选查询和对应框。
3. refer_adapter 产生原始候选表示。
4. BCEM 使用同一候选索引下的特征和框进行背景增强。
5. CRS 使用增强后的候选表示产生最终分数。

实现不直接修改本次前向中的预测框、检测类别或候选索引。联合训练仍可通过共享骨干间接改变后续的候选生成分布，因此评价评分贡献时需要固定候选实验。

REFHead 的构造默认关闭 BCEM，关闭时不分配其参数，并保留原有初始化流。现有 AerialVG 配置中直接提供 BCEM 参数并默认启用，可通过 model.head.enable_bcem=False 运行基线。当前 AerialVG 采用 REFHead；GTMHead 没有在本次改动中接入 BCEM。

## 7 完整数据流与实现接口

~~~mermaid
flowchart TD
    I["输入图像与完整文本"] --> ENC["BEiT-3 图文编码与融合"]
    ENC --> FPN["FPN 多尺度视觉特征"]
    ENC --> T["文本特征 T"]
    FPN --> DEC["候选生成 Decoder"]
    DEC --> BOX["预测候选框：几何采样时停止梯度"]
    DEC --> Q0["候选查询"]
    Q0 --> BIND["可选 Grab 筛选与绑定：同步查询和框"]
    BOX --> BIND
    BIND --> ADP["refer_adapter"]
    ADP --> Q["原始候选表示 q"]
    BIND --> SAMP["8 个方向 × 4 个尺度：10%、15%、20%、25%"]
    FPN --> PROJ["最高分辨率检测特征：256 → 128"]
    PROJ --> SAMP
    MASK["图像尺寸与视觉 mask"] --> SAMP
    SAMP --> U["32 个框外背景片段：每片段 9 点"]
    SAMP --> Z["空间位置与有效性"]
    U --> NODE["32 个背景节点证据"]
    Z --> NODE
    U --> PAIR["24 个相邻连接：端点、差分、位置与间隔"]
    Z --> PAIR
    PAIR --> GRU["共享 GRU：每方向独立状态、由近到远"]
    GRU --> TR["24 个转移证据"]
    NODE --> E["56 个证据与有效性 mask"]
    TR --> E
    E --> READ["文本 Query 读取背景 Key、Value：追加零 Value 空证据"]
    T --> READ
    TM["文本 padding mask"] --> READ
    READ --> EP["有效文本位置平均：背景表示 e"]
    EP --> G["门控：q、e、有效文本平均"]
    Q --> G
    T --> G
    TM --> G
    G --> FUSE["残差融合：q_enh = q + 0.1 × g × P(e)"]
    EP --> FUSE
    Q --> FUSE
    FUSE --> CRS["CRS 指代评分"]
    T --> CRS
    TM --> CRS
    CRS --> SCORE["最终候选分数"]
~~~

设批量大小为 $B$、候选数为 $N$、文本长度为 $L$，接口如下：

| 数据 | 形状 | 对应实现 |
| --- | --- | --- |
| 原候选表示 | $B\times N\times256$ | refer_adapter 输出 |
| 归一化预测框 | $B\times N\times4$ | 相对有效图像的 cxcywh |
| 文本特征 | $B\times L\times256$ | lan_embedding 输出 |
| 背景片段 | $B\times N\times8\times4\times128$ | _sample_regions |
| 位置数值 | $B\times N\times8\times4\times11$ | _sample_regions |
| 转移证据 | $B\times N\times8\times3\times128$ | _encode_evidence |
| 合并证据 | $B\times N\times56\times128$ | _encode_evidence |
| 背景表示 | $B\times N\times128$ | _read_background |
| 背景使用权重 | $B\times N\times1$ | gate |
| 增强候选表示 | $B\times N\times256$ | forward 返回值 |

实现文件：

- [背景增强模块](propvg/models/heads/background_context.py)
- [训练与推理接入](propvg/models/heads/REFHead.py)
- [现有 AerialVG 配置](configs/aerialvg/PropVG-aerialvg.py)
- [回归测试](tools/test_bcem.py)

模块返回增强候选表示与已停止梯度的诊断信息。REFHead 的 extra_dict 中可读取 bcem_context、bcem_gate、bcem_residual_ratio、bcem_valid_nodes、bcem_valid_transitions。默认不保留完整词语注意力矩阵；若开展词语读取可视化，应另行增加按需导出接口。

### 7.1 默认参数与计算处理

| 参数 | 默认值 |
| --- | --- |
| 背景维度 | 128 |
| 注意力头数 | 4 |
| 主方向数 | 8 |
| 采样尺度 | 0.10、0.15、0.20、0.25 |
| 每片段采样点数 | 9 |
| 有效点数阈值 | 5 |
| 子角度偏移 | −15°、0°、15° |
| 转移编码 | 两层 MLP＋共享单层 GRUCell |
| 背景使用权重 | 可学习标量 |
| 全局融合强度 | 固定 0.1 |
| 候选分块大小 | 32 |
| 分块激活重计算 | 开启 |
| 注意力 dropout | 0 |

默认完整模块包含 449,153 个可训练参数；关闭 GRU 为 350,081 个，只保留背景节点为 248,193 个，固定背景使用权重为 1 为 366,976 个。这些数量按独立 BCEM 模块统计，不包含原 PropVG 参数。

完整模块的背景读取复杂度与 $BNL(56+1)D$ 成正比，另含视觉投影、采样、转移编码和门控开销。当前配置每个候选读取 $32\times9=288$ 个采样点。

实现先对整幅特征图进行一次投影，再分块处理候选，避免为每个候选复制图像特征。采用 PyTorch scaled_dot_product_attention；其具体后端由设备和输入条件决定。训练时对分块计算使用非重入激活 checkpoint，以减少保留的中间激活，代价是反向时重新计算。仅分块而不重计算，不能保证训练阶段全部注意力激活的总占用随块大小下降。

GPU 峰值显存、吞吐和推理延迟需要在目标硬件上实测，不能由参数数量或理论注意力矩阵大小直接替代。

### 7.2 启动与消融配置

在安装了本项目依赖的 Python 环境中，从仓库根目录执行：

~~~bash
# 单 GPU 训练，现有配置默认启用 BCEM
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py --launcher none

# 两 GPU 训练
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29500 tools/train.py \
  configs/aerialvg/PropVG-aerialvg.py --launcher pytorch

# 从同结构的 AerialVG 基线权重初始化，新增 BCEM 使用随机初始化
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --finetune-from /path/to/aerialvg-baseline.pth

# PropVG 基线，关闭 BCEM
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.enable_bcem=False

# 仅背景节点与位置编码
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.bcem.use_transitions=False

# 保留相邻连接，关闭 GRU
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.bcem.use_gru=False

# 固定背景使用权重为 1
PYTHONPATH=. python tools/train.py configs/aerialvg/PropVG-aerialvg.py \
  --cfg-options model.head.bcem.learnable_gate=False

# 推理使用同一 BCEM 配置与训练后的 checkpoint
PYTHONPATH=. python tools/test.py configs/aerialvg/PropVG-aerialvg.py \
  --load-from /path/to/bcem-checkpoint.pth --launcher none

# CPU 回归测试
PYTHONPATH=. python tools/test_bcem.py
~~~

旧基线权重用于新增分支的初始化时，采用 finetune-from；resume-from 用于继续同结构、同配置的 BCEM 训练并恢复优化器。直接将新增参数结构当成旧优化器的无缝续训并不成立。BCEM 参数已合并到现有 AerialVG 配置，原训练脚本可直接使用；进行基线对照时关闭 model.head.enable_bcem。

## 8 实验验证设计

### 8.1 递进消融与贡献归因

| 版本 | 要回答的问题 | 当前实现状态 |
| --- | --- | --- |
| PropVG | 基线能力 | model.head.enable_bcem=False |
| 扩展框外区域特征＋文本读取 | 单纯增加背景观测是否有效 | 建议增加的实验对照 |
| 32 个背景节点＋位置编码＋文本读取 | 四尺度空间采样是否有效 | use_transitions=False |
| 上述设置＋24 个相邻连接，无 GRU | 相邻连接是否有额外价值 | use_gru=False |
| 完整 BCEM | 顺序累积是否有额外价值 | 默认配置 |
| 完整 BCEM，固定 $g_i=1$ | 自适应使用权重是否必要 | learnable_gate=False |
| 相同数量的非相邻连接 | 收益是否依赖相邻结构 | 建议增加的实验对照 |

从节点版本增加转移证据时，证据数量与参数数量同时发生变化。应进一步使用容量接近的对照或相同数量的非相邻连接，排除“更多参数或更多证据”这一解释。各版本统一数据划分、候选数、视觉来源、训练预算及四个尺度，并报告实际参数规模和计算开销。

对于顺序扰动，需要明确扰动对象。仅重排附带原坐标的证据集合，并不等同于破坏空间结构；改变 GRU 输入顺序、错配视觉内容与坐标、改变相邻连接分别检验不同机制。

### 8.2 固定候选实验

从同一基线生成并固定候选框、候选表示和对应图文特征，冻结骨干及候选生成端，以一致预算训练评分端。分别报告：

1. 候选集合对 GT 的覆盖率；
2. 已有候选满足目标 IoU 阈值时的条件选择准确率；
3. 最终 Top-1、Top-5 与框 mIoU。

这能够区分评分改善与候选生成改善。由于 BEiT-3 特征已经受文本条件影响，缓存应以图像和表达组成的样本为单位。

### 8.3 样本分组与诊断

重点分析场景环境关系、辅助对象关系、单一与多个关系，以及外观相似候选具有不同或相似背景的情况。还应单独检查图像边缘附近、有效背景证据较少的样本。

关键词规则只适合作为分组起点，应结合人工抽查并报告样本数；不同组可以重叠。诊断信息用于统计背景使用权重、实际残差比例和证据有效率，不应将门控数值当作正确性解释。

当前 BCEM 配置继承既有数据设置：训练使用 vg_val_odvg.jsonl，验证和测试均使用 vg_test_odvg.jsonl。该设置没有在本次改动中重划分。正式选择超参数及报告结果前，应确定独立的验证集和最终测试集，避免在最终测试集上选取设置。

## 9 适用边界与预期贡献

本研究预期形成一种候选相关、文本条件化的背景增强机制，为 CRS 补充场景与辅助对象证据。其贡献能否进一步归因于相邻转移和顺序累积，应由受控实验决定。如果原始背景节点与位置编码已经达到相同效果，则应将贡献收敛为背景读取与候选增强，而不将转移编码作为核心创新。

方法具有以下边界：四尺度采样是稀疏观测，可能漏掉采样区域之外的辅助对象；较大框外距离降低了直接获取极近邻细节的能力；最高分辨率 FPN 来自已有骨干特征，不能恢复骨干已经丢失的细节；文本平均读取最终形成单个向量，不能保证严格满足多条件逻辑组合；背景描述相同的候选仍可能需要目标外观与其他关系线索进行区分。

实现已通过采样几何、padding 不变性、无效背景恒等退化、文本屏蔽、梯度传播、缺失连接状态清空、分块一致性、激活重计算、消融开关以及 REFHead 训练／推理接入的 CPU 回归验证。这些验证说明实现满足设定的计算行为，不代表已获得航拍指代性能提升。当前会话未进行完整数据集训练，也未测量 GPU 开销。

## 参考资料

1. Dai M., et al. PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination. ICCV, 2025. [论文](https://arxiv.org/html/2509.04833v1)，[官方代码](https://github.com/Dmmm1997/PropVG)。
2. Liu J., et al. AerialVG: A Challenging Benchmark for Aerial Visual Grounding by Exploring Positional Relations. arXiv:2504.07836. [论文](https://arxiv.org/html/2504.07836v3)。
