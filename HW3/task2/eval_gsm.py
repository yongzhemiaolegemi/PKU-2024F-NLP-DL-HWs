import json
direct_dir = 'direct_gsm8k.json'
cot_dir = 'cot_gsm8k_new.json'
few_shot_dir = 'cot_gsm8k_few_shot.json'
reflect_dir = 'cot_gsm8k_reflection.json'
rag_dir = 'gsm_rag_cot.json'
from tqdm import tqdm
def eval_math(list, type_str):
    count_pass = 0
    count_fail = 0
    if type_str == "direct":
        search_key = "ds_answer"
    if type_str == "cot":
        search_key = "direct_ds_answer"
    if type_str == "few_shot":
        search_key = "direct_ds_answer"
    if type_str == "reflect":
        search_key = "direct_reflect_ds_answer"
    if type_str == "rag":
        search_key = "direct_ds_answer"
    
    for i, item in enumerate(tqdm(list)):
        model_answer = item[search_key]
        gt_answer = item['direct_answer']
        if model_answer == gt_answer:
            count_pass += 1
        else:
            count_fail += 1
    return count_pass, count_fail


with open(direct_dir, 'r') as f:
    direct_list = json.load(f)
with open(cot_dir, 'r') as f:
    cot_list = json.load(f)
with open(few_shot_dir, 'r') as f:
    few_shot_list = json.load(f)
with open(reflect_dir, 'r') as f:
    reflect_list = json.load(f)
with open(rag_dir, 'r') as f:
    rag_list = json.load(f)
total = len(direct_list)
total_pass, total_fail = eval_math(direct_list, "direct")
print(f"Direct GSM8k pass/fail rate:")
print(total_pass/total, total_fail/total)
total_pass, total_fail = eval_math(cot_list, "cot")
print(f"Cot GSM8k pass/fail rate:")
print(total_pass/total, total_fail/total)
total_pass, total_fail = eval_math(few_shot_list, "few_shot")
print(f"Few Shot GSM8k pass/fail rate:")
print(total_pass/total, total_fail/total)
total_pass, total_fail = eval_math(reflect_list, "reflect")
print(f"Reflect GSM8k pass/fail rate:")
print(total_pass/total, total_fail/total)
total_pass, total_fail = eval_math(rag_list, "rag")
print(f"RAG GSM8k pass/fail rate:")
print(total_pass/total, total_fail/total)