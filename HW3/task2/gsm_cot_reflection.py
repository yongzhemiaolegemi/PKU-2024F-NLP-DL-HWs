# load_dataset
from datasets import load_dataset
# ds = load_dataset("openai/gsm8k","main")["test"]
# ls_ds = list(ds)
from tqdm import tqdm
import re
import json
# def get_direct_answer(text):
#     match = re.search(r'####\s*(.*)', text)

#     if match:
#         answer = match.group(1)  # Extract the number after '####'
#         return answer
#     else:
#         print("No direct answer found")
#         print('text:', text)
#         exit()
# output_list = []
# for i, item in enumerate(tqdm(ls_ds)):
#     answer = item['answer']
#     direct_answer = get_direct_answer(answer)
#     item['direct_answer'] = direct_answer
#     output_list.append(item)
# gt_dir = "/home/dna-paradiam/chenxinyu/rag/gsm8k.json"
# dir = "/home/dna-paradiam/chenxinyu/rag/cot_gsm8k.json"
# with open(gt_dir, 'r') as f:
#     gt_list = json.load(f)
# with open(dir, 'r') as f:
#     output_list = json.load(f)
# new_output_list = []
# for i, item in enumerate(tqdm(output_list)):
#     gt_item = gt_list[i]
#     new_item = gt_item
#     new_item['full_ds_answer'] = item['answer']
#     new_item['direct_ds_answer'] = item['direct_answer']
#     new_output_list.append(new_item)
# output_dir = 'cot_gsm8k_new.json'
# with open(output_dir, 'w') as f:
#     json.dump(new_output_list, f, indent=4)
# exit()

output_dir = 'cot_gsm8k_new.json'
import json
# with open(output_dir, 'w') as f:
#     json.dump(output_list, f, indent=4)
with open(output_dir, 'r') as f:
    output_list = json.load(f)
api_key = "sk-28d48ddaca3f42459642d6788cbbc9f6"
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI


def get_deepseek_generation(prompt,answer=None, prompt_2 = None):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    if prompt_2 is None:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
    else:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": prompt_2},
            ],
        )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content
def filter_answer(answer):
    matches = re.findall(r'\[(.*?)\]', answer)
    if len(matches) == 0:
        return "none"
    return matches[-1]
count = 0
total_count = len(output_list)
model_output_dir = 'cot_gsm8k_reflection.json'
# model_output = []
with open(model_output_dir, 'r') as f:
    model_output = json.load(f)
now_index = len(model_output)
for i, item in enumerate(tqdm(output_list)):
    if i < now_index:
        continue
    system_prompt = "Solve the following question and provide the answer directly in [ ]. The answer should be a number without any additional text.\n"
    system_prompt_cot = "Solve the following question by thinking step by step and provide the answer in the end in [ ]. The final answer should be a number without any additional text.\n"
    reflection_prompt_cot = "Reflect on your reasoning and final response. Check if there are any mistakes, misinterpretations, or potential improvements in the reasoning or final answer. If you find any issues, revise your answer and provide the corrected thinking process as well as result in the end in [ ]."
    original_answer = item['full_ds_answer']
    prompt = item['question']
    full_prompt = system_prompt_cot + prompt
    deepseek_answer = get_deepseek_generation(full_prompt, original_answer, reflection_prompt_cot) 
    get_ds_answer = filter_answer(deepseek_answer)
    print(get_ds_answer)
    
    gt_answer = item['direct_answer']
    item['full_reflect_ds_answer'] = deepseek_answer
    item['direct_reflect_ds_answer'] = get_ds_answer
    model_output.append(item)
    with open(model_output_dir, 'w') as f:
        json.dump(model_output, f, indent=4)
    
    if get_ds_answer == gt_answer:
        count += 1
    else:
        print("GT:", gt_answer)
        print("DS:", get_ds_answer)
        print('question:', prompt)
print(f"Accuracy: {count}/{total_count} = {count/total_count:.2f}")
