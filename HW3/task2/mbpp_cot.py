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
    
output_dir = 'mbpp.json'
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
model_output_dir = "cot_mbpp.json"

for i, item in enumerate(tqdm(output_list)):
    code = item['code']
    def_code = item['def_code']
    text = item['text']
    prompt = "Repeat the following function definition and complete the code inside the function. Output the entire Python code as a plain string that can be directly executed using `exec()`. Do not add any extra content."
    prompt_cot = '''Repeat the following function definition and complete the code inside the function. Output the entire Python code as a plain string that can be directly executed using `exec()`. You're required to think step by step. For each step, explain your reasoning in code comments. You can structure your thinking and explanations in any way that is helpful in code comments.
    **Make sure the thinking process is written as comments, so that it doesn't interfere with the execution of the code.**
    '''



    full_prompt = text + '\n' + prompt_cot  + def_code 
    output = get_deepseek_generation(full_prompt)
    # full_code = def_code+'\n'+output
    full_code = remove_block(output)
    test_cases = item['test_list']
    print(full_code)

    item['full_code'] = full_code
    model_oupput.append(item)
    with open(model_output_dir, 'w') as f:
        json.dump(model_oupput, f, indent=4)

    
    try:
        # Execute the function definition code
        exec(full_code)
        
        # Flag to track if all test cases pass
        all_tests_passed = True

        # Execute all test cases
        for test_case in test_cases:
            try:
                exec(test_case)
            except Exception as e:
                print(f'Error in test case at index {i}: {test_case}')
                all_tests_passed = False
                break  # Exit the loop on first error in the test case

        # If all test cases passed, increment the count
        if all_tests_passed:
            count += 1

    except Exception as e:
        print(f'Error executing function definition at index {i}: {e}')

print(f'Number of successful test cases: {count}')

    