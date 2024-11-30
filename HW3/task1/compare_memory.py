import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig  # 支持量化

def measure_gpu_memory_usage(model, tokenizer, input_text, max_new_tokens, use_kv_cache, num_repeats):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    baseline_memories = []
    final_memories = []
    peak_memories = []

    for _ in range(num_repeats):
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        past_key_values = None if use_kv_cache else None

        # 清空缓存，准备测量
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 测试前的显存基线
        baseline_memory = torch.cuda.memory_allocated(device)

        # 推理
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids, use_cache=use_kv_cache, past_key_values=past_key_values)
                input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                if use_kv_cache:
                    past_key_values = outputs.past_key_values

        # 测试后的显存
        final_memory = torch.cuda.memory_allocated(device)
        peak_memory = torch.cuda.max_memory_allocated(device)

        baseline_memories.append(baseline_memory)
        final_memories.append(final_memory)
        peak_memories.append(peak_memory)

    # 计算平均值
    avg_baseline = sum(baseline_memories) / num_repeats
    avg_final = sum(final_memories) / num_repeats
    avg_peak = sum(peak_memories) / num_repeats

    return avg_baseline, avg_final, avg_peak

def test_quantization_memory(model_name, input_text, max_new_tokens, num_repeats=5):
    results = {}
    device_map = "auto"

    # 基线模型
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results['baseline'] = measure_gpu_memory_usage(model, tokenizer, input_text, max_new_tokens, use_kv_cache=False, num_repeats=num_repeats)

    # KV-cache 优化
    results['kv_cache'] = measure_gpu_memory_usage(model, tokenizer, input_text, max_new_tokens, use_kv_cache=True, num_repeats=num_repeats)

    # fp16 量化
    quant_config_fp16 = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=False,  # 确保不启用 FP32 转换
        bnb_4bit_compute_dtype=torch.float16    # 设置为 torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map,quantization_config=quant_config_fp16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results['fp16'] = measure_gpu_memory_usage(model, tokenizer, input_text, max_new_tokens, use_kv_cache=True, num_repeats=num_repeats)

    # int8 量化
    quant_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True,bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results['int8'] = measure_gpu_memory_usage(model, tokenizer, input_text, max_new_tokens, use_kv_cache=True, num_repeats=num_repeats)

    return results

# 配置实验
model_name = "gpt2"  # 示例模型
input_text = "Once upon a time"  # 输入文本
max_new_tokens = 50  # 最大生成的 tokens 数
num_repeats = 10  # 重复次数

# 执行实验
memory_results = test_quantization_memory(model_name, input_text, max_new_tokens, num_repeats)

# 打印结果
print("显存使用（MB）：")
for config, (baseline, final, peak) in memory_results.items():
    print(f"{config}: 初始 {baseline / 1024**2:.2f} MB, 结束 {final / 1024**2:.2f} MB, 峰值 {peak / 1024**2:.2f} MB (平均值，重复 {num_repeats} 次)")
