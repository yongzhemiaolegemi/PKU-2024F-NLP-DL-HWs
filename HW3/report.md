# NLP HW3 陈昕宇 2200017735

## Task1

### 1.1

本实验测试了在生成任务中不同优化策略对推理效率的影响，包括基线模型、KV-cache 优化、FP16 量化和 INT8 量化。我们针对 3 组不同生成长度（tokens=50、250、500）的情况，分别测量了每秒生成的 token 数量（吞吐量），并重复 10 次取平均值，最终记录了各策略的性能表现。

#### 实验数据

以下是不同生成长度下的推理效率（吞吐量，单位：tokens/秒）：

| Token 数 | Baseline | KV-cache | FP16   | INT8   |
| -------- | -------- | -------- | ------ | ------ |
| 50       | 82.51    | 79.61    | 106.06 | 111.20 |
| 250      | 91.94    | 86.57    | 108.70 | 107.04 |
| 500      | 94.52    | 82.15    | 111.15 | 113.44 |

------

#### 实验发现

1. **Baseline（基线模型）**
   - 基线模型在短序列生成（tokens=50）和中序列生成（tokens=250）中表现稳定，随着序列长度增加，吞吐量有所提升（从 82.51 增加到 94.52 tokens/秒）。
   - 但由于未使用任何优化，推理效率较低。
2. **KV-cache 优化**
   - KV-cache 的吞吐量在短序列和中序列生成时低于基线模型（如 tokens=50 时为 79.61 tokens/秒，低于基线的 82.51 tokens/秒）。
   - 对于长序列生成（tokens=500），其吞吐量（82.15 tokens/秒）依然不及基线模型。
3. **FP16 量化**
   - FP16 的吞吐量在所有序列长度下均优于基线和 KV-cache 优化（如 tokens=50 时为 106.06 tokens/秒，tokens=500 时达到 111.15 tokens/秒）。
4. **INT8 量化**
   - INT8 的吞吐量在所有情况下都高于其他优化方法（如 tokens=50 时为 111.20 tokens/秒，tokens=500 时达到 113.44 tokens/秒）。
   - INT8 的权重存储和计算占用显存最小，使得 GPU 能更高效地利用其计算资源。

------

#### 量化的影响分析

1. **速度与显存的权衡**
   - FP16 和 INT8 量化显著提升了推理速度，同时有效减少了显存占用：
     - **FP16** 在不显著降低模型精度的前提下，加速了推理。
     - **INT8** 提供了更极致的显存优化和速度提升，但可能在极端低精度情况下导致推理性能或输出质量下降。
2. **KV-cache 的特殊性**
   - 虽然 KV-cache 在理论上可以通过缓存键值对减少计算量，但在小模型和短序列生成任务中，其初始化和存储操作的开销掩盖了潜在的加速效果。
   - KV-cache 的优势更适合长序列生成和大模型（如 GPT-3 或 LLaMA-13B）。

### 1.2

本实验针对生成长度为 **max tokens=50** 的场景，测量了不同优化策略对显存使用的影响。通过 10 次重复测量，记录了初始显存、结束显存以及峰值显存的平均值，以分析优化方法（Baseline、KV-cache、FP16、INT8）在显存使用上的特点和差异。

------

#### 实验结果

| 配置     | 初始显存 (MB) | 结束显存 (MB) | 峰值显存 (MB) | 测试次数 |
| -------- | ------------- | ------------- | ------------- | -------- |
| Baseline | 326.72        | 327.55        | 329.16        | 10       |
| KV-cache | 329.21        | 329.41        | 332.27        | 10       |
| FP16     | 423.57        | 423.62        | 429.19        | 10       |
| INT8     | 197.04        | 197.10        | 512.77        | 10       |

------

#### 实验发现

1. **Baseline（基线模型）**
   - **初始显存**：326.72 MB，与其他配置相比显存占用最小，仅加载模型权重和基础输入数据。
   - **峰值显存**：329.16 MB，推理过程中显存增加较少，主要来自中间计算图的显存需求。
   - **结论**：Baseline 显存使用最稳定，但由于未采用优化方法，推理效率较低。
2. **KV-cache 优化**
   - **初始显存**：329.21 MB，略高于 Baseline，因需分配缓存存储键值对。
   - **峰值显存**：332.27 MB，略高于 Baseline，推理中缓存的键值对显著增加了显存需求。
   - **结论**：KV-cache 在短序列生成任务中对显存影响有限，但没有充分体现其优化潜力（适用于长序列生成）。
3. **FP16 量化**
   - **初始显存**：423.57 MB，比 Baseline 和 KV-cache 高出约 30%，由于使用 FP16 存储权重，其显存需求高于非量化模型。
   - **峰值显存**：429.19 MB，推理过程中占用较多显存，主要因为 FP16 激活计算需要额外的显存开销。
   - **结论**：FP16 显存需求高于 Baseline，但相比 INT8 显存峰值更稳定，适合在计算资源较充足时使用。
4. **INT8 量化**
   - **初始显存**：197.04 MB，显存使用最少，INT8 显著降低了权重存储需求。
   - **峰值显存**：512.77 MB，峰值显存大幅增加，可能由于激活或中间计算需要频繁进行 INT8 与 FP32/FP16 的转换。
   - **结论**：INT8 初始显存最低，适合显存受限的硬件，但峰值显存波动较大，可能在实际任务中引发显存溢出问题。

------

#### 优化方法的对比分析

1. **显存占用对比**：
   - 初始显存：INT8 < Baseline < KV-cache < FP16
   - 峰值显存：Baseline < KV-cache < FP16 < INT8
   - 分析：
     - INT8 对初始显存优化最明显，但峰值显存波动较大。
     - FP16 的显存使用相对平衡，峰值显存略高但可控。
     - KV-cache 在短序列任务中显存开销较小。

### 2.贪心解码时间对比

作为bonus 实现了prefix caching

#### 时间数据

- **无KV缓存的黄金标准贪心解码**：7.02秒
- **Prefix Caching的定制化贪心解码**：4.19秒
- **Vanilla定制化贪心解码**：5.39秒

#### 分析

1. **Prefix Caching vs 无KV缓存**：
   - 使用**Prefix Caching**的定制化贪心解码显著比**无KV缓存**的黄金标准解码更快，时间减少了约 **2.83秒**，即约 **40%** 的时间节省。Prefix Caching通过缓存已计算的前缀（部分序列），避免了每次生成时重复计算这些部分，从而提高了推理效率。
2. **Prefix Caching vs Vanilla定制化贪心解码**：
   - 与普通的定制化贪心解码（Vanilla）相比，Prefix Caching也提供了较明显的优化，节省了约 **1.19秒**，即约 **22%** 的时间。尽管Vanilla解码相较于无KV缓存（黄金标准）已经有所优化，但Prefix Caching进一步通过缓存推理过程中的中间状态来提升性能。
3. **模型优化的影响**：
   - 使用KV缓存（或Prefix Caching）对长文本生成任务尤为重要，能够大幅度提高推理速度，尤其是当生成的文本依赖于较长的上下文时。相比之下，**Vanilla定制化解码**没有这种优化，虽然仍比黄金标准有所改进，但效果不如Prefix Caching明显。

#### 结论

通过实验结果可以得出，**Prefix Caching**显著加速了解码过程，尤其是在生成较长文本时，可以避免重复计算，提高模型推理效率。相比之下，**Vanilla定制化解码**虽然有所优化，但性能仍不如Prefix Caching。



## Task2

### 2.1 RAG系统实现

我们使用了广泛使用的文本编码模型all-MiniLM-L6-v2作为embedding model

在具体实现上，对于gsm8k任务，我们提取了lighteval/MATH的训练数据集作为知识库，将question+answer合并提取得到条目并获得相应的embedding。对于mbpp任务，我们提取了iamtarun/python_code_instructions_18k_alpaca作为知识库，处理方式类似。

在inference过程中，我们匹配question和知识库中条目的相似程度，提取出top 3相似的条目作为补充知识

### 2.2 RAG和多种不同prompting技巧上的性能提升比较

#### MBPP

实现细节：

对于直接输出答案，我们要求模型重复函数声明并完成补全

对于cot，我们要求模型在thinking step by step的过程中补全函数，思考过程注释掉

对于few shot，few shot=3

对于reflect 我们要求模型在CoT结果上进行修正，修正思路注释掉

pass==通过所有测试样例

fail==一部分测试样例没通过

error==函数未通过编译



| Method        | Pass Rate | Fail Rate | Error Rate |
| ------------- | --------- | --------- | ---------- |
| Direct MBPP   | 0.562     | 0.436     | 0.002      |
| Cot MBPP      | 0.382     | 0.324     | 0.294      |
| Few Shot MBPP | 0.564     | 0.43      | 0.006      |
| Reflect MBPP  | 0.382     | 0.324     | 0.294      |
| RAG MBPP      | 0.538     | 0.438     | 0.024      |

从表格中可以看到，不同的 prompting 技巧对 MBPP 任务的性能产生了明显影响：

- **Direct MBPP** 方法的通过率为 **56.2%**，表现较为稳定，失败率为 **43.6%**，错误率较低（**0.2%**）。该方法较为简单，直接输出函数补全，适用于简单任务。
- **CoT MBPP** 方法的通过率较低，只有 **38.2%**，失败率为 **32.4%**，错误率较高（**29.4%**）。尽管 CoT 可以通过引导模型逐步思考，但在此任务中没有显著提高性能，反而可能因为过多的思考步骤导致误差增加。
- **Few Shot MBPP** 方法的表现与 Direct MBPP 相似，通过率为 **56.4%**，失败率为 **43%**，错误率为 **0.6%**，略高于 Direct 方法。这表明在少量示例指导下，模型仍能保持较好的性能，适合当没有足够的训练数据时使用。
- **Reflect MBPP** 方法的表现与 CoT 方法类似，均为 **38.2%** 的通过率，失败率 **32.4%**，错误率 **29.4%**，表明反思并修正 CoT 结果并未带来显著改进。
- RAG方法和Few Shot方法类似，但是可能由于不同数据集格式不同，导致函数编译失败的概率有所上升

综合来看， **Direct MBPP** 、Few Shot MBPP和RAG方法 在通过率和错误率方面表现较为理想，尤其适合简单任务和有少量训练示例的情况；而 **CoT** 和 **Reflect** 在 MBPP 任务中的提升较小，甚至可能因思考过程过于复杂导致性能下降。

### GSM8k

实现细节：

对于直接输出答案，我们要求模型直接输出数值结果

对于cot，我们要求模型在thinking step by step后输出数值结果

对于few shot，few shot=3

对于reflect 我们要求模型在CoT结果上进行修正，之后输出数值结果

| Method         | Pass Rate | Fail Rate |
| -------------- | --------- | --------- |
| Direct GSM8k   | 0.3904    | 0.6096    |
| Cot GSM8k      | 0.9212    | 0.0788    |
| Few Shot GSM8k | 0.9030    | 0.0970    |
| Reflect GSM8k  | 0.8870    | 0.1130    |
| RAG GSM8k      | 0.8082    | 0.1918    |

### 结果分析：

- **Direct GSM8k** 方法的 **Pass Rate** 为 0.3904，**Fail Rate** 为 0.6096，表现较差，原因可能是缺乏足够的推理过程，模型只能依赖其原始能力生成结果。
- **CoT GSM8k** 通过逐步思考方式显著提高了正确率，**Pass Rate** 达到 0.9212，**Fail Rate** 仅为 0.0788，表明通过详细推理步骤，模型能够更准确地解决问题。
- **Few Shot GSM8k** 的性能略低于 CoT，但仍表现优秀，**Pass Rate** 为 0.9030。该方法通过少量示例引导模型，虽然没有 CoT 那么详细的推理过程，但足以帮助模型找到正确答案。
- **Reflect GSM8k** 的 **Pass Rate** 为 0.8870，略低于 Few Shot，说明反思方法能够一定程度上修正 CoT 生成的错误，但可能不如原始的 CoT 推理过程来得直接有效。
- **RAG GSM8k** 方法的 **Pass Rate** 为 0.8082，**Fail Rate** 为 0.1918。通过结合知识库的 3 条最相关条目，RAG 方法提升了模型的表现，但相比 CoT 和 Few Shot，仍有一定差距。RAG 在处理复杂问题时可能存在知识库内容和问题之间的关联性不足，导致表现相对较弱。

总的来说，**CoT** 方法在 GSM8k 任务中表现最好，能够显著提高模型的准确性，而 **Few Shot** 和 **RAG** 也能有效提升性能，尤其在需要外部知识时，RAG 提供了有力的补充。

### Appendix

Few Shot 提高了代码水平的样例：

Write a function to find the first duplicate element in a given array of integers.

direct

```python
def find_first_duplicate(nums):
    num_set = set()
    no_duplicate = -1
    for i in range(len(nums)):
        if nums[i] in num_set:
            return nums[i]
        else:
            num_set.add(nums[i])

    return no_duplicate

```

few shot

```python
def find_first_duplicate(nums):
    # Step 1: Initialize a set to keep track of seen numbers
    seen = set()
    
    # Step 2: Iterate through each number in the array
    for num in nums:
        # Step 3: Check if the number is already in the set
        if num in seen:
            # Step 4: If it is, return the number as it is the first duplicate
            return num
        # Step 5: If it is not, add the number to the set
        seen.add(num)
    
    # Step 6: If no duplicates are found, return -1
    return -1

# Example usage:
# nums = [2, 3, 4, 5, 3, 2]
# print(find_first_duplicate(nums))  # Output: 3
```

Cot提高了数学水平的样例：

direct

question: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?

direct_answer: 68

cot_answer: 

First, calculate the cost of the glasses that are full price:

Kylar wants to buy 16 glasses, so half of them (8 glasses) will be full price.

8 full-price glasses * $5 per glass = $<<8*5=40>>40.



Next, calculate the cost of the glasses that are discounted:\nThe remaining 8 glasses will be at 60% of the full price.

60% of $5 is $5 * 0.6 = $<<5*0.6=3>>3 per glass.

So, 8 discounted glasses * $3 per glass = $<<8*3=24>>24.



Finally, add the cost of the full-price glasses and the discounted glasses:

$40 for full-price glasses + $24 for discounted glasses = $<<40+24=64>>64.



[64]
