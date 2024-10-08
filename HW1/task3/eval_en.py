import torch
import torch.nn as nn
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
sentences = []
with open('ws353simrel\wordsim353_sim_rel\wordsim_similarity_goldstandard.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        words = line.split()
        sentences.append([words[0], words[1], words[2]])
        # print(words)
class Word2VecCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecCBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    
    def forward(self, context_words):

        embeds = self.embeddings(context_words)  # (batch_size, context_size, embedding_dim)
        context_embedding = torch.sum(embeds, dim=1)  # (batch_size, embedding_dim)
        

        out = self.linear(context_embedding)  # (batch_size, vocab_size)
        return out

import torch.optim as optim


import json
dir = "train_data_en.json"
with open(dir, "r", encoding='utf-8') as f:
    data = json.load(f)
training_data = data["training_data"]
word2idx = data["word2idx"]
embedding_dim = 128
vocab_size = len(word2idx)


model = Word2VecCBOW(vocab_size, embedding_dim)


checkpoint = torch.load('en_cbow_model_and_optimizer_20.pth')
# model.load_state_dict(checkpoint['model_state_dict'])


model.eval()  
def evaluate(data, word2idx, model):

    sim = []
    for pair in data:
        word1 = pair[0]
        word2 = pair[1]
        score = pair[2]
        if word1 in word2idx and word2 in word2idx:
            idx1 = torch.tensor([word2idx[word1]], dtype=torch.long)
            idx2 = torch.tensor([word2idx[word2]], dtype=torch.long)
            vec1 = model.embeddings(idx1)
            vec2 = model.embeddings(idx2)
            loss = (torch.cosine_similarity(vec1, vec2)*5+5 - float(score))**2
            sim.append(loss.item())
    return sim
sim = evaluate(sentences, word2idx, model)
print(sum(sim)/len(sim))