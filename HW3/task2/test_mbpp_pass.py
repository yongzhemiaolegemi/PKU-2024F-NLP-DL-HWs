import json
dir_1 = "direct_mbpp.json"
dir_2 = "cot_mbpp.json"
from tqdm import tqdm
def get_new(list, new_dir):
    model_output_dir = new_dir
    model_output = []
    count = 0
    for i, item in enumerate(tqdm(list)):
        full_code = item['full_code']
        test_cases = item['test_list']
        try:
            # Execute the function definition code
            print(full_code)

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
                    item['pass'] = "False"
                    break  # Exit the loop on first error in the test case

            # If all test cases passed, increment the count
            if all_tests_passed:

                count += 1
                item['pass'] = "True"

        except Exception as e:
            print(f'Error executing function definition at index {i}: {e}')
            item['pass'] = "Error"
        model_output.append(item)
        with open(model_output_dir, 'w') as f:
            json.dump(model_output, f, indent=4)
    print(count)






with open(dir_1, 'r') as f:
    direct_list = json.load(f)
with open(dir_2, 'r') as f:
    cot_list = json.load(f)
new_dir_1 = "direct_mbpp_new.json"
new_dir_2 = "cot_mbpp_new.json"
get_new(direct_list, new_dir_1)
get_new(cot_list, new_dir_2)