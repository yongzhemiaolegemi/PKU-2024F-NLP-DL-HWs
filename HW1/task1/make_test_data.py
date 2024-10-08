import random
def get_list(len, list_len):
    len = int(len)
    list_len = int(list_len)
    return [[random.randint(1, 100) for i in range(len)] for j in range(list_len)]

test_1_1e1 = get_list(5, 10/5)
test_1_1e2 = get_list(5, 100/5)
test_1_1e3 = get_list(5, 1000/5)
test_1_1e4 = get_list(5, 10000/5)
test_1_1e5 = get_list(5, 100000/5)
test_1_1e6 = get_list(5, 1000000/5)
test_1_1e7 = get_list(5, 10000000/5)
test_2_1e1 = get_list(10/10, 10)
test_2_1e2 = get_list(100/10, 10)
test_2_1e3 = get_list(1000/10, 10)
test_2_1e4 = get_list(10000/10, 10)
test_2_1e5 = get_list(100000/10, 10)
test_2_1e6 = get_list(1000000/10, 10)
test_2_1e7 = get_list(10000000/10, 10)
import json
dir = "test_data.json"
with open(dir, "w") as f:
    json.dump({
        "test_1_1e1": test_1_1e1,
        "test_1_1e2": test_1_1e2,
        "test_1_1e3": test_1_1e3,
        "test_1_1e4": test_1_1e4,
        "test_1_1e5": test_1_1e5,
        "test_1_1e6": test_1_1e6,
        "test_1_1e7": test_1_1e7,
        "test_2_1e1": test_2_1e1,
        "test_2_1e2": test_2_1e2,
        "test_2_1e3": test_2_1e3,
        "test_2_1e4": test_2_1e4,
        "test_2_1e5": test_2_1e5,
        "test_2_1e6": test_2_1e6,
        "test_2_1e7": test_2_1e7,
    }, f, indent=4)