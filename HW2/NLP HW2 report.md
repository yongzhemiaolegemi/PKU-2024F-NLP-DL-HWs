# NLP HW2 report

陈昕宇 2200017735

## Task2

batch_size = 16 epoch=3 其他参数均为默认

具体图片见imgs

三个模型均表现出性能上：

agnews>res>acl

这可能是因为acl的label数量大于res和agnews

在衡量对比上，三个模型训练后的整体性能

仅在acl上，scibert大于bert和roberta，bert和roberta性能相近。这是因为 SciBERT 是专门针对科学文献和学术领域的文本进行预训练的。相比之下，BERT 和 RoBERTa 主要在通用语料上进行预训练，因此在处理领域特定的术语、概念和句法结构时，SciBERT 能够更好地理解和建模科学文本中的细微差异。

在agnews和res数据集上三个模型表现大致相当，这是因为这两个数据集不需要特定的学科专业知识，因此scibert没有展现出预训练带来的优势

另外，观察到training loss和任务性能的相关性

## Task3

### GPU占用分析

### **1. 估算不使用 PEFT（Adapter）时的 GPU 内存占用**



考虑一个具有 **30 亿（3B）** 参数的模型，以下是对 GPU 内存占用的估算。

**模型参数：**

- **总参数量：** \( 3 \times 10^9 \) 参数。
- **每个参数的存储：** 使用 16 位浮点（FP16）格式，每个参数占用 **2 字节**。

$$
\text{参数内存} = 3 \times 10^9 \text{ 参数} \times 2 \text{ 字节/参数} = 6 \text{ GB}
$$

**梯度：**

- 反向传播时，需要存储与参数同等大小的梯度。

$$
\text{梯度内存} = 6 \text{ GB}
$$

**优化器状态：**

- 以 Adam 优化器为例，需要为每个参数存储两个额外的状态（动量和二阶矩），每个占用 2 字节。

$$
\text{优化器内存} = 3 \times 10^9 \text{ 参数} \times 4 \text{ 字节/参数} = 12 \text{ GB}
$$

**激活值：**

- 激活值的内存占用与模型的深度、批量大小和序列长度有关。假设激活内存约为参数内存的 **1 倍**。

$$
\text{激活内存} \approx 6 \text{ GB}
$$

**总内存占用（不使用 PEFT）：**

$$
\begin{align*}
\text{总内存} &= \text{参数内存} + \text{梯度内存} + \text{优化器内存} + \text{激活内存} \\
&= 6 \text{ GB} + 6 \text{ GB} + 12 \text{ GB} + 6 \text{ GB} \\
&= 30 \text{ GB}
\end{align*}
$$

**估计的 GPU 内存需求：** **约 30 GB**

---

### **2. 估算使用 Adapter 时的 GPU 内存占用**

假设在该模型的每个 Transformer 层中插入 Adapter，并且每层出现两次 Adapter。我们只微调 Adapter，其余参数保持冻结。

**Adapter 的参数量计算**

假设：

- **隐藏维度（hidden\_size，\( H \)）：** 通过模型参数量估计。
- **Adapter 维度（adapter\_size，\( A \)）：** 64
- **Transformer 层数（\( L \)）：** 24

**估计隐藏维度 \( H \) 和层数 \( L \)**

总参数量主要由 Transformer 层贡献，可近似为：

$$
\text{参数量} \approx L \times (12H^2)
$$

其中，\( 12H^2 \) 是每层的参数量。

解方程：

$$
3 \times 10^9 = 24 \times (12H^2) \\
\Rightarrow 3 \times 10^9 = 288H^2 \\
\Rightarrow H^2 = \frac{3 \times 10^9}{288} \\
\Rightarrow H^2 \approx 10,416,666.67 \\
\Rightarrow H \approx 3,230
$$

因此，隐藏维度 \( H \approx 3,230 \)。

**计算 Adapter 的参数量**

**单个 Adapter 的参数量：**

$$
\begin{align*}
\text{降维层参数} &= (H \times A) + A = (3,230 \times 64) + 64 = 206,720 + 64 = 206,784 \\
\text{升维层参数} &= (A \times H) + H = (64 \times 3,230) + 3,230 = 206,720 + 3,230 = 209,950 \\
\text{单个 Adapter 参数总量} &= 206,784 + 209,950 = 416,734
\end{align*}
$$

**每层的 Adapter 参数总量（每层有两个 Adapter）：**

$$
\text{每层 Adapter 参数量} = 416,734 \times 2 = 833,468
$$

**整个模型的 Adapter 参数总量（共 24 层）：**

$$
\text{Adapter 总参数量} = 833,468 \times 24 = 20,003,232
$$

#### **GPU 内存占用计算（只微调 Adapter）**

**参数内存：**

- **可训练参数（Adapter）：**

$$
\text{Adapter 参数内存} = 20,003,232 \text{ 参数} \times 2 \text{ 字节/参数} \approx 40 \text{ MB}
$$

- **冻结参数（其余的模型参数）：**

$$
\text{冻结参数内存} = (3 \times 10^9 - 20,003,232) \times 2 \text{ 字节/参数} \approx 5,959.99 \text{ MB}
$$

- **总参数内存：**

$$
\text{参数内存} = 40 \text{ MB} + 5,959.99 \text{ MB} \approx 6,000 \text{ MB} = 6 \text{ GB}
$$

**梯度：**

- **只需为可训练参数（Adapter）存储梯度：**

$$
\text{梯度内存} = 40 \text{ MB}
$$

**优化器状态：**

- **只需为可训练参数（Adapter）存储优化器状态：**

$$
\text{优化器内存} = 20,003,232 \text{ 参数} \times 4 \text{ 字节/参数} \approx 80 \text{ MB}
$$

**激活值：**

- 激活值的内存占用与不使用 PEFT 时相同，因为前向传播仍需经过整个模型。

$$
\text{激活内存} \approx 6 \text{ GB}
$$

**总内存占用（使用 Adapter）：**

$$
\begin{align*}
\text{总内存} &= \text{参数内存} + \text{梯度内存} + \text{优化器内存} + \text{激活内存} \\
&= 6 \text{ GB} + 40 \text{ MB} + 80 \text{ MB} + 6 \text{ GB} \\
&= 12 \text{ GB} + 120 \text{ MB} \\
&\approx 12.12 \text{ GB}
\end{align*}
$$

---

### **3. PEFT（Adapter）可微调参数占比的重新估计**

**总参数量：**

$$
\text{总参数量} = 3 \times 10^9 \text{ 参数}
$$

**Adapter 可微调参数量：**

$$
\text{Adapter 参数量} = 20,003,232 \text{ 参数}
$$

**可微调参数占比：**

$$
\text{占比} = \frac{\text{Adapter 参数量}}{\text{总参数量}} = \frac{20,003,232}{3 \times 10^9} \approx 0.6668\%
$$

---

### **4. GPU 内存节省**

**不使用 PEFT 的总内存：** 约 **30 GB**

**使用 Adapter 的总内存：** 约 **12.12 GB**

**节省的内存：**

$$
\text{节省的内存} = 30 \text{ GB} - 12.12 \text{ GB} = 17.88 \text{ GB}
$$

**节省的内存百分比：**

$$
\text{节省的百分比} = \frac{17.88 \text{ GB}}{30 \text{ GB}} \times 100\% \approx 59.6\%
$$

---

### **总结**

1. **不使用 PEFT（Adapter）时，微调 3B 参数的模型需要约 30 GB 的 GPU 内存。**

2. **使用 Adapter 后，GPU 内存占用降至约 12.12 GB，节省了约 59.6\% 的内存。**

3. **Adapter 的可微调参数占总参数量的约 0.6668\%。**

---

以下为adapter的一个实现，上述分析以此为准

``` python
class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x + residual  # Add residual connection
```

### 实验结果

见imgs

使用peft微调后，模型性能相比全参数微调在acl上下降明显（约4%），在res上下降不显著（约0.5%），在agnews上几乎没有变化

该结果与以上三个任务的难度相关性强，初步推测模型的peft微调性能下降和任务难度有很强的相关性

另外，观察到无论哪个任务，peft的training loss均比全参数高约0.1

