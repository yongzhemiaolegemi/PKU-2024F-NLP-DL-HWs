import json
direct_mbpp_dir = "direct_mbpp_new.json"
few_shot_mbpp_dir = "cot_mbpp_few_shot.json"
with open(direct_mbpp_dir, 'r') as f:
    direct_mbpp_list = json.load(f)
with open(few_shot_mbpp_dir, 'r') as f:
    few_shot_mbpp_list = json.load(f)
for i, item in enumerate(direct_mbpp_list):
    direct_pass = item['pass']
    few_shot_pass = few_shot_mbpp_list[i]['pass']
    if direct_pass !="True" and few_shot_pass == "True":
        print(i)
        print(direct_mbpp_list[i])
        print(few_shot_mbpp_list[i])
        exit()