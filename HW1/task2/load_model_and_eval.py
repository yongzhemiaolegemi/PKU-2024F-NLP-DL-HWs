import torch
import torch.nn as nn

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

# 定义超参数
import json
dir = "train_data.json"
with open(dir, "r", encoding='utf-8') as f:
    data = json.load(f)
training_data = data["training_data"]
word2idx = data["word2idx"]
embedding_dim = 128
vocab_size = len(word2idx)


model = Word2VecCBOW(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

checkpoint = torch.load('cbow_model_and_optimizer.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


model.eval()  


