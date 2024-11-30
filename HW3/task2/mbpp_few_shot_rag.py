def find_def(code):

    # 找到第一个 'def' 的位置
    def_pos = code.find('def')

    if def_pos != -1:
        # 从 'def' 开始向后查找第一个冒号的位置
        colon_pos = code.find(':', def_pos)
        
        if colon_pos != -1:
            # 截取从开头到冒号之前的内容
            result = code[:colon_pos+1]  # 包括冒号
            return result
        else:
            print("not found")
            print('code:', code)
            exit()
    else:
        print("not found")
        print('code:', code)
        exit()
import json
from datasets import load_dataset
# ds = load_dataset('google-research-datasets/mbpp', 'full')['test']
# ls_ds = list(ds)
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
    
output_dir = 'mbpp_rag.json'
from tqdm import tqdm
# output_list = []
# for i, item in enumerate(tqdm(ls_ds)):
#     code = item['code']
#     def_code = find_def(code)
    
#     item['def_code'] = def_code
#     output_list.append(item)
# with open(output_dir, 'w') as f:
#     json.dump(output_list, f, indent=4)
with open(output_dir, 'r') as f:
    output_list = json.load(f)
def remove_block(code):
    if code.startswith('```python'):
        code = code.strip('```python')
    if code.endswith('```'):
        code = code.strip('```')
    return code
count = 0
model_oupput = []
total_count = len(output_list)
model_output_dir = "mbpp_rag_cot.json"
# with open(model_output_dir, 'r') as f:
#     model_oupput = json.load(f)
model_oupput = []
now_index = len(model_oupput)
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
import random
import signal
TIME_LIMIT = 2
# Define a timeout handler that raises an exception
def timeout_handler(signum, frame):
    raise TimeoutError("Test case execution exceeded time limit.")
count = 0
for i, item in enumerate(tqdm(output_list)):
    if i < now_index:
        continue
    code = item['code']
    def_code = item['def_code']
    text = item['text']
    prompt = "Repeat the following function definition and complete the code inside the function. Output the entire Python code as a plain string that can be directly executed using `exec()`. Do not add any extra content."
    prompt_cot = '''Repeat the following function definition and complete the code inside the function. Output the entire Python code as a plain string that can be directly executed using `exec()`. You're required to think step by step. For each step, explain your reasoning in code comments. You can structure your thinking and explanations in any way that is helpful in code comments.
    **Make sure the thinking process is written as comments, so that it doesn't interfere with the execution of the code.**
    '''

    

    # select 3 random items from output_list
    # few_shot_items = random.sample(output_list, 3)
    # example_1  = few_shot_items[0]
    # example_2  = few_shot_items[1]
    # example_3  = few_shot_items[2]
    # question_1 = example_1['text']
    # answer_1 = example_1["code"]
    # question_2 = example_2['text']
    # answer_2 = example_2["code"]
    # question_3 = example_3['text']
    # answer_3 = example_3["code"]
    question_1 = item['question_1']
    answer_1 = item['answer_1']
    question_2 = item['question_2']
    answer_2 = item['answer_2']
    question_3 = item['question_3']
    answer_3 = item['answer_3']
    shot_prompt_full = shot_prompt.format(question_1=question_1, answer_1=answer_1, question_2=question_2, answer_2=answer_2, question_3=question_3, answer_3=answer_3)



    full_prompt = shot_prompt_full+text + '\n' + prompt_cot  + def_code 
    output = get_deepseek_generation(full_prompt)
    # full_code = def_code+'\n'+output
    full_code = remove_block(output)
    test_cases = item['test_list']
    print(full_code)

    item['full_code'] = full_code


    
    try:
        # Execute the function definition code
        namespace = {}
        exec(full_code, namespace)
        
        # Flag to track if all test cases pass
        all_tests_passed = True

        # Execute all test cases
        for test_case in test_cases:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(TIME_LIMIT)
                exec(test_case, namespace)
                item['pass'] = "True"
                signal.alarm(0)
            except Exception as e:
                print(f'Error in test case at index {i}: {test_case}')
                print(f'Error message: {e}')
                all_tests_passed = False
                item['pass'] = "False"
                break  # Exit the loop on first error in the test case
        del namespace
        # If all test cases passed, increment the count
        if all_tests_passed:
            count += 1

    except Exception as e:
        print(f'Error executing function definition at index {i}: {e}')
        item['pass'] = "Error"

    model_oupput.append(item)
    with open(model_output_dir, 'w') as f:
        json.dump(model_oupput, f, indent=4)

print(f'Number of successful test cases: {count}')

    