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


import json
dir = "train_data.json"
with open(dir, "r", encoding='utf-8') as f:
    data = json.load(f)
training_data = data["training_data"]
word2idx = data["word2idx"]
embedding_dim = 128
vocab_size = len(word2idx)



if torch.cuda.is_available():
    device = torch.device('cuda')
model = Word2VecCBOW(vocab_size, embedding_dim)
model.to(device)
nn.init.normal_(model.embeddings.weight, mean=0, std=0.01)
nn.init.constant_(model.linear.weight, 0)


criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=1e-2)
import random
def generate_batches(data, batch_size):
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        context_batch = torch.tensor([item[1] for item in batch], dtype=torch.long)
        target_batch = torch.tensor([item[0] for item in batch], dtype=torch.long)
        assert target_batch.max().item() < vocab_size, "Target index out of range!"
        # print(context_batch, target_batch)

        yield context_batch, target_batch


epochs = 100
batch_size = 1
from tqdm import tqdm
for epoch in range(epochs):
    loss_1k = 0
    global_step = 0


    for context_batch, target_batch in generate_batches(training_data, batch_size):
        # if global_step % 100==0:
        #     print(context_batch, target_batch)
        
        optimizer.zero_grad()

        context_batch = context_batch.to(device)
        
        output = model(context_batch)

        
        # if global_step % 100==0:
        #     print(output, target_batch)
        target_batch = target_batch.to(device)
        loss = criterion(output, target_batch)
        
        
        loss_1k += loss.item()


        loss.backward()


        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

        optimizer.step()
    
        global_step += 1
        # if global_step % 100==0:
        #     print(loss.item())
        if global_step % 1000 == 0:
            print(f"Epoch {epoch + 1}, Step {global_step}, Loss {loss_1k / 1000}")
            loss_1k = 0
    if (epoch+1) % 5 == 0:

        path = f'cbow_model_and_optimizer_{epoch+1}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)


        
