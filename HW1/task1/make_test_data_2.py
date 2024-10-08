import random
def make_random_string(length):
    # only alphabet with lower case
    return ''.join(random.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(length))

test_1e1 = make_random_string(10)
test_1e2 = make_random_string(100)
test_1e3 = make_random_string(1000)
test_1e4 = make_random_string(10000)
test_1e5 = make_random_string(100000)
test_1e6 = make_random_string(1000000)
test_1e7 = make_random_string(10000000)
import json
dir = "test_data_2.json"
with open(dir, "w") as f:
    json.dump({
        "test_1e1": test_1e1,
        "test_1e2": test_1e2,
        "test_1e3": test_1e3,
        "test_1e4": test_1e4,
        "test_1e5": test_1e5,
        "test_1e6": test_1e6,
        "test_1e7": test_1e7,
    }, f, indent=4)