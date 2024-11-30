import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig  # 支持int8量化

def measure_throughput(model, tokenizer, input_text, max_new_tokens, use_kv_cache):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    past_key_values = None if use_kv_cache else None

    # 清空缓存，准备测量
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 开始计时
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    # 推理
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, use_cache=use_kv_cache, past_key_values=past_key_values)
            input_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            if use_kv_cache:
                past_key_values = outputs.past_key_values
    end_time.record()

    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time) / 1000  # 秒
    throughput = max_new_tokens / elapsed_time
    return throughput



# 配置实验
model_name = "gpt2"  # 示例模型
input_text = "Once upon a time"  # 输入文本
max_new_tokens = 500  # 最大生成的 tokens 数
num_repeats = 10  # 每种配置重复的次数

# 修改 test_with_quantization 函数，增加重复测试逻辑
def test_with_quantization_average(model_name, input_text, max_new_tokens, num_repeats):
    results = {}
    device_map = "auto"

    # 测试基线模型（无量化）
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    throughput_list = [
        measure_throughput(model, tokenizer, input_text, max_new_tokens, use_kv_cache=False)
        for _ in range(num_repeats)
    ]
    results['baseline'] = sum(throughput_list) / num_repeats

    # KV-cache 优化
    throughput_list = [
        measure_throughput(model, tokenizer, input_text, max_new_tokens, use_kv_cache=True)
        for _ in range(num_repeats)
    ]
    results['kv_cache'] = sum(throughput_list) / num_repeats

    # fp16 量化
    quant_config_fp16 = BitsAndBytesConfig(
        llm_int8_enable_fp32_cpu_offload=False,  # 确保不启用 FP32 转换
        bnb_4bit_compute_dtype=torch.float16    # 设置为 torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device_map, quantization_config=quant_config_fp16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    throughput_list = [
        measure_throughput(model, tokenizer, input_text, max_new_tokens, use_kv_cache=True)
        for _ in range(num_repeats)
    ]
    results['fp16'] = sum(throughput_list) / num_repeats

    # int8 量化
    quant_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True,bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    throughput_list = [
        measure_throughput(model, tokenizer, input_text, max_new_tokens, use_kv_cache=True)
        for _ in range(num_repeats)
    ]
    results['int8'] = sum(throughput_list) / num_repeats

    return results

# 执行实验
results = test_with_quantization_average(model_name, input_text, max_new_tokens, num_repeats)

# 打印结果
print("推理效率（吞吐量，tokens/秒）：")
for config, throughput in results.items():
    print(f"{config}: {throughput:.2f} tokens/秒 (平均值，重复 {num_repeats} 次)")

