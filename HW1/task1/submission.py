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