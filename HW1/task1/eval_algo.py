def flatten_list(nested_list: list):
    output_list = []
    for list in nested_list:
        output_list.extend(list)
    return output_list

    
    ... # your code here


def char_count(s: str):
    char_count_dict = {}
    for char in s:
        if char in char_count_dict:
            char_count_dict[char] += 1
        else:
            char_count_dict[char] = 1
    return char_count_dict
    ... # your code here.
import json
dir = "test_data_2.json"
with open(dir, "r") as f:
    test_data = json.load(f)
test_round = 10
running_time = {}
for i in range(test_round):

    for key, value in test_data.items():
        import time

        begin = time.time()
        print(f"Running test {key}...")
        print("Flattening list...")
        char_count(value)
        print("Done.")
        ... # your program here
        end = time.time()

        time = end - begin
        if key in running_time:
            running_time[key].append(time)
        else:
            running_time[key] = [time]

for key, value in running_time.items():
    print(f"Test {key} average running time = {sum(value) / len(value)}")
          
        

