import random
with open('eng_jpn.txt', 'r', encoding='utf-8') as file:
    #split 8:1:1
    lines = file.readlines()
    random.shuffle(lines)
    total = len(lines)
    train_lines = lines[:int(total*0.8)]
    dev_lines = lines[int(total*0.8):int(total*0.9)]
    test_lines = lines[int(total*0.9):]
    print(len(train_lines), len(dev_lines), len(test_lines))
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open('dev.txt', 'w', encoding='utf-8') as f:
        f.writelines(dev_lines)
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_lines)