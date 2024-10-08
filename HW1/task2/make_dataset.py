import numpy as np
import random
from collections import defaultdict
import jieba

sentences = []
with open('train.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        words = line.split()
        sentences.append(words[0])
tokenized_sentences = []
for sentence in sentences:
    words = jieba.lcut(sentence)  
    tokenized_sentences.append(words)



vocab = set([word for sentence in tokenized_sentences for word in sentence])

vocab.add('<UNK>')
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}


def generate_training_data(tokenized_sentences, window_size):
    training_data = []
    for sentence in tokenized_sentences:
        sentence_length = len(sentence)
        for idx, word in enumerate(sentence):
            context_window = list(range(max(0, idx - window_size), min(sentence_length, idx + window_size + 1)))
            context_window.remove(idx)  
            context_window_index = []
            for context_idx in context_window:
                context_window_index.append(word2idx[sentence[context_idx]])
            
            training_data.append((word2idx[word], context_window_index))
            # check
            print(f"Center word: {word}, Context word: {[idx2word[idx] for idx in context_window_index]}")
    return training_data

training_data = generate_training_data(tokenized_sentences, window_size=5)
# print(training_data)  
import json
dir = "train_data.json"
with open(dir, "w", encoding='utf-8') as f:
    json.dump({
        "training_data": training_data,
        "word2idx": word2idx,
        "idx2word": idx2word
    }, f, indent=4, ensure_ascii=False)