import json
import signal
import signal
TIME_LIMIT = 2
# Define a timeout handler that raises an exception
def timeout_handler(signum, frame):
    raise TimeoutError("Test case execution exceeded time limit.")
def test_code(code, test_cases):
    is_pass = 'false'
    full_code = code
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
                is_pass = 'true'
                signal.alarm(0)
            except Exception as e:
                # print(f'Error in test case at index {i}: {test_case}')
                # print(f'Error message: {e}')
                is_pass = 'false'
                break  # Exit the loop on first error in the test case
        del namespace
        # If all test cases passed, increment the count


    except Exception as e:
        # print(f'Error executing function definition at index {i}: {e}')
        is_pass = 'error'
    return is_pass
from tqdm import tqdm
def eval_code(output_list, type_str):
    count_pass = 0
    count_error = 0
    count_false = 0
    total_count = len(output_list)
    for i, item in enumerate(tqdm(output_list)):
        code = item['full_code']
        test_list = item['test_list']
        is_pass = test_code(code, test_list)
        if is_pass == 'true':
            count_pass += 1
        elif is_pass == 'false':
            count_false += 1
        else:
            count_error += 1
    print(f"Type: {type_str}")
    print(f"Pass rate: {count_pass}/{total_count}")
    print(f"False rate: {count_false}/{total_count}")
    print(f"Error rate: {count_error}/{total_count}")
    return count_pass, count_false, count_error
    
direct_mbpp_dir = "/home/dna-paradiam/chenxinyu/rag/direct_mbpp_new.json"
cot_mbpp_dir = "/home/dna-paradiam/chenxinyu/rag/cot_mbpp_new.json"
few_shot_mbpp_dir = "/home/dna-paradiam/chenxinyu/rag/cot_mbpp_few_shot.json"
reflect_mbpp_dir = "/home/dna-paradiam/chenxinyu/rag/cot_mbpp_reflection.json"
rag_mbpp_dir = "/home/dna-paradiam/chenxinyu/rag/mbpp_rag_cot.json"
with open(direct_mbpp_dir, 'r') as f:
    direct_mbpp_list = json.load(f)
with open(cot_mbpp_dir, 'r') as f:
    cot_mbpp_list = json.load(f)
with open(few_shot_mbpp_dir, 'r') as f:
    few_shot_mbpp_list = json.load(f)
with open(reflect_mbpp_dir, 'r') as f:
    reflect_mbpp_list = json.load(f)
with open(rag_mbpp_dir, 'r') as f:
    rag_mbpp_list = json.load(f)
total = len(direct_mbpp_list)
count_pass_direct, count_false_direct, count_error_direct = eval_code(direct_mbpp_list, "Direct MBPP")
count_pass_cot, count_false_cot, count_error_cot = eval_code(cot_mbpp_list, "Cot MBPP")
count_pass_few_shot, count_false_few_shot, count_error_few_shot = eval_code(few_shot_mbpp_list, "Few Shot MBPP")
count_pass_reflect, count_false_reflect, count_error_reflect = eval_code(reflect_mbpp_list, "Reflect MBPP")
count_pass_rag, count_false_rag, count_error_rag = eval_code(rag_mbpp_list, "RAG MBPP")
print(f"Total: {total}")
print(f"Direct MBPP pass/fail/error rate:")
print(count_pass_direct/total, count_false_direct/total, count_error_direct/total)
print(f"Cot MBPP pass/fail/error rate:")
print(count_pass_cot/total, count_false_cot/total, count_error_cot/total)
print(f"Few Shot MBPP pass/fail/error rate:")
print(count_pass_few_shot/total, count_false_few_shot/total, count_error_few_shot/total)
print(f"Reflect MBPP pass/fail/error rate:")
print(count_pass_reflect/total, count_false_reflect/total, count_error_reflect/total)
print(f"RAG MBPP pass/fail/error rate:")
print(count_pass_rag/total, count_false_rag/total, count_error_rag/total)

