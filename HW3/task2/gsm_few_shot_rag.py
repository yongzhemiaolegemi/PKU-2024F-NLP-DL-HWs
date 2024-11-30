

shot_prompt = '''You will be provided with a few examples of questions and answers as few-shot samples. Please pay close attention to the format of these examples, as the format of the output you are required to generate might differ from the examples.The following are the question-answer pairs (few-shot examples). After these examples, you will receive a specific task with an output requirement.\n
Examples:
Question:\n
{question_1}\n
Answer:\n
{answer_1}\n
Question:\n
{question_2}\n
Answer:\n
{answer_2}\n
Question:\n
{question_3}\n
Answer:\n
{answer_3}\n
Task:\n
'''
# load_dataset
from datasets import load_dataset
ds = load_dataset("openai/gsm8k","main")["test"]
ls_ds = list(ds)
from tqdm import tqdm
import re
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
output_dir = 'gsm_rag.json'
import json
# with open(output_dir, 'w') as f:
#     json.dump(output_list, f, indent=4)
with open(output_dir, 'r') as f:
    output_list = json.load(f)
api_key = "sk-28d48ddaca3f42459642d6788cbbc9f6"
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI


def get_deepseek_generation(prompt):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
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
model_output_dir = 'gsm_rag_cot.json'
# with open(model_output_dir, 'r') as f:
#     model_output = json.load(f)
# now_index = len(model_output)
now_index = 0
model_output = []
import random
for i, item in enumerate(tqdm(output_list)):
    if i < now_index:
        continue
    system_prompt = "Solve the following question and provide the answer directly in [ ]. The answer should be a number without any additional text.\n"
    system_prompt_cot = "Solve the following question by thinking step by step and provide the answer in the end in [ ]. The final answer should be a number without any additional text.\n"
    # few_shot_items = random.sample(output_list, 3)
    # question_1 = few_shot_items[0]['question']
    # answer_1 = few_shot_items[0]['answer']
    # question_2 = few_shot_items[1]['question']
    # answer_2 = few_shot_items[1]['answer']
    # question_3 = few_shot_items[2]['question']
    # answer_3 = few_shot_items[2]['answer']
    question_1 = item['question_1']
    answer_1 = item['answer_1']
    question_2 = item['question_2']
    answer_2 = item['answer_2']
    question_3 = item['question_3']
    answer_3 = item['answer_3']
    full_shot_prompt = shot_prompt.format(question_1=question_1, answer_1=answer_1, question_2=question_2, answer_2=answer_2, question_3=question_3, answer_3=answer_3)


    prompt = item['question']
    full_prompt = full_shot_prompt + system_prompt_cot + prompt
    deepseek_answer = get_deepseek_generation(full_prompt)
    get_ds_answer = filter_answer(deepseek_answer)
    print(get_ds_answer)
    gt_answer = item['direct_answer']

    item['full_ds_answer'] = deepseek_answer
    item['direct_ds_answer'] = get_ds_answer
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
