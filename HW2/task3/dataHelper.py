from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import json
def get_texts_and_labels_restaurant_sup(data_samples, sep_token):
    texts = [
        sample['sentence'].replace(sample['term'], f"{sep_token} {sample['term']}")
        for sample in data_samples.values()
    ]

    labels = [
        1 if sample['polarity'] == "positive" else
        2 if sample['polarity'] == "neutral" else
        0
        for sample in data_samples.values()
    ]

    return texts, labels


def get_texts_and_labels_acl_sup(data_samples):
    texts = [
        sample["text"] for sample in data_samples
    ]
    unique_labels = list(sorted(set(sample['label'] for sample in data_samples)))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_id[sample['label']] for sample in data_samples]

    return texts, labels
def get_acl_dataset(first_name, second_name, sep_token):
    train_dir = "ACL/train.jsonl"
    with open(train_dir, 'r') as f:
        train_data = [json.loads(line) for line in f]
    test_dir = "ACL/test.jsonl"
    with open(test_dir, 'r') as f:
        test_data = [json.loads(line) for line in f]
    train_texts, train_labels = get_texts_and_labels_acl_sup(train_data)
    test_texts, test_labels = get_texts_and_labels_acl_sup(test_data)
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    test_dataset = Dataset.from_dict({
        'text': test_texts,
        'label': test_labels
    })
    return train_dataset, test_dataset


def get_agnews_dataset(first_name, second_name, sep_token):
    dataset = load_dataset('ag_news', split='test')
    
    # 使用 `train_test_split` 以 9:1 比例划分训练集和测试集，设定随机种子为 2022
    split_dataset = dataset.train_test_split(test_size=0.1, seed=2022)
    
    # 构造 DatasetDict 对象
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    return train_dataset, test_dataset


def get_restaurant_dataset(first_name, second_name, sep_token):
    train_dir = "SemEval14-res/train.json"
    with open(train_dir, 'r') as f:
        train_data = json.load(f)
    test_dir = "SemEval14-res/test.json"
    with open(test_dir, 'r') as f:
        test_data = json.load(f)
    train_texts, train_labels = get_texts_and_labels_restaurant_sup(train_data, sep_token)
    test_texts, test_labels = get_texts_and_labels_restaurant_sup(test_data, sep_token)
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    test_dataset = Dataset.from_dict({
        'text': test_texts,
        'label': test_labels
    })
    return train_dataset, test_dataset


def get_laptop_dataset(first_name, second_name, sep_token):
    train_dir = "SemEval14-laptop/train.json"
    with open(train_dir, 'r') as f:
        train_data = json.load(f)
    test_dir = "SemEval14-laptop/test.json"
    with open(test_dir, 'r') as f:
        test_data = json.load(f)
    train_texts, train_labels = get_texts_and_labels_restaurant_sup(train_data, sep_token)
    test_texts, test_labels = get_texts_and_labels_restaurant_sup(test_data, sep_token)
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    test_dataset = Dataset.from_dict({
        'text': test_texts,
        'label': test_labels
    })
    return train_dataset, test_dataset


def get_single_dataset(dataset_name, sep_token):
    first_name, second_name = dataset_name.split("_")[0], dataset_name.split("_")[1]

    if first_name =="acl":
        train_dataset, test_dataset = get_acl_dataset(first_name, second_name, sep_token)


        
    if first_name=="agnews":
        train_dataset, test_dataset = get_agnews_dataset(first_name, second_name, sep_token)
        


    if first_name =="restaurant":
        train_dataset, test_dataset = get_restaurant_dataset(first_name, second_name, sep_token)


    if first_name =="laptop":
        train_dataset, test_dataset = get_laptop_dataset(first_name, second_name, sep_token)

    if second_name =="fs":
        train_dataset =  train_dataset.shuffle(seed=42).select(range(32))
        test_dataset = test_dataset.shuffle(seed=42).select(range(32))


    return train_dataset, test_dataset

def rearrange_labels(datasets):
    now_label_add_idx = 0
    output_datasets = []
    for dataset in datasets:
        unique_labels = list(sorted(set(dataset['label'])))

        label_map = {old_label: new_label + now_label_add_idx for new_label, old_label in enumerate(unique_labels)}
        

        dataset = dataset.map(lambda example: {"label": label_map[example['label']]})
        output_datasets.append(dataset)

        now_label_add_idx += len(unique_labels)
    return output_datasets


def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str, the name of the dataset
    sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
    '''
    if isinstance(dataset_name, str):
        train_dataset, test_dataset = get_single_dataset(dataset_name, sep_token)
    
    if isinstance(dataset_name, list):
        train_datasets, test_datasets = [], []
        for dataset_name in dataset_name:
            train_dataset, test_dataset = get_single_dataset(dataset_name, sep_token)
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        train_datasets = rearrange_labels(train_datasets)
        test_datasets = rearrange_labels(test_datasets)
        train_dataset = concatenate_datasets(train_datasets)
        test_dataset = concatenate_datasets(test_datasets)


    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    return dataset

        


if __name__ == '__main__':
    # 调用 get_dataset 函数并加载数据集
    dataset = get_dataset(['restaurant_fs', 'laptop_fs', 'acl_fs'], "<sep>")

    # 检查数据集中是否包含 'train' 和 'test' 集合
    assert 'train' in dataset, "Dataset does not contain a 'train' split."
    assert 'test' in dataset, "Dataset does not contain a 'test' split."

    # 检查 'train' 和 'test' 数据集的字段是否正确
    for split in ['train', 'test']:
        assert 'text' in dataset[split].column_names, f"'text' column is missing in the {split} dataset."
        assert 'label' in dataset[split].column_names, f"'label' column is missing in the {split} dataset."

    # 输出一些样本，检查 'sep_token' 是否已正确添加，以及情感标签是否符合预期
    print("Sample from the train set:")
    print("Text:", dataset['train']['text'][0])
    print("Label:", dataset['train']['label'][0])

    print("\nSample from the test set:")
    print("Text:", dataset['test']['text'][0])
    print("Label:", dataset['test']['label'][0])

    # print label range
    print("\nLabel range:", set(dataset['train']['label']))
    print("Label range:", set(dataset['test']['label']))
    # print dataset len
    print("\nTrain set size:", len(dataset['train']))
    print("Test set size:", len(dataset['test']))

    print("\nAll tests passed. Dataset is processed successfully.")

    