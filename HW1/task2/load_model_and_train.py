import torch
import torch.nn as nn
import jieba
import torch
import torch.nn as nn
import torch.nn.functional as F
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



cbow_model = Word2VecCBOW(vocab_size, embedding_dim)


checkpoint = torch.load('cbow_model_and_optimizer_50.pth')
cbow_model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


cbow_model.eval()  
sentences = []
labels = []
def sentence2idx(words, word2idx):
    idx_list = []
    for word in words:
        if word in word2idx:
            idx = word2idx[word]
        else:
            idx = word2idx['<UNK>']
        idx_list.append(idx)
    return idx_list

def get_dataset(dir):
    with open(dir, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            words = line.split()
            sentences.append("".join(words[:-1]))
            labels.append(words[-1])

    tokenized_sentences = []
    for sentence in sentences:
        words = jieba.lcut(sentence)
        idx_list = sentence2idx(words, word2idx)  
        if(len(idx_list)<5):
            idx_list = idx_list +[word2idx['<UNK>']]*(5-len(idx_list))
        tokenized_sentences.append(idx_list)
    return tokenized_sentences, labels
tokenized_sentences, labels = get_dataset('train.txt')
eval_tokenized_sentences, eval_labels = get_dataset('dev.txt')
test_tokenized_sentences, test_labels = get_dataset('test.txt')

def evaluate(model, tokenized_sentences, labels):
    correct = 0
    total = 0
    for i in range(len(tokenized_sentences)):
        sentence = torch.tensor(tokenized_sentences[i], dtype=torch.long).unsqueeze(0).to(device)
        sentence_vector = cbow_model.embeddings(sentence)
        label = torch.tensor([int(labels[i])], dtype=torch.long).to(device)
        output = model(sentence_vector)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    return correct / total

class ConvNetForText(nn.Module):
    def __init__(self, embedding_dim, window_size, num_filters, k):

        super(ConvNetForText, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=window_size)

        self.fc = nn.Linear(num_filters, k)
    
    def forward(self, x):

        x = x.permute(0, 2, 1)  
        

        x = self.conv(x)  
        x = F.relu(x)  

        x = F.max_pool1d(x, kernel_size=x.size(2))  
        x = x.squeeze(2)  

        x = self.fc(x)  
        return x

model  = ConvNetForText(embedding_dim, 3, 64, 4)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
cbow_model = cbow_model.to(device)
prev_acc = 0
for epoch in range(epochs):
    total_loss = 0
    global_step = 0
    for i in range(len(tokenized_sentences)):
        sentence = torch.tensor(tokenized_sentences[i], dtype=torch.long).unsqueeze(0).to(device)
        sentence_vector = cbow_model.embeddings(sentence)

        label = torch.tensor([int(labels[i])], dtype=torch.long).to(device)
        optimizer.zero_grad()
        # print(sentence_vector.size())
        output = model(sentence_vector)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        global_step += 1
        if global_step % 1000 == 0:
            print(f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item()}")
            acc = evaluate(model, eval_tokenized_sentences, eval_labels)
            print(f"Accuracy on dev set: {acc}")
    # save model
    path = f'conv_model_{epoch + 1}.pth'
    torch.save(model.state_dict(), path)
    # apply early stopping
    acc = evaluate(model, eval_tokenized_sentences, eval_labels)
    acc_test = evaluate(model, test_tokenized_sentences, test_labels)
    print("------epoch {}------".format(epoch))
    print(f"Accuracy on test set: {acc_test}")
    print("--------------------")
    if acc < prev_acc:
        break
    prev_acc = acc
