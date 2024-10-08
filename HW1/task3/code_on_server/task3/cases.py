import torch
import torch.nn as nn
from tqdm import tqdm
from janome.tokenizer import Tokenizer
from torch.utils.data import Dataset, DataLoader
import os
import random
import wandb


tokenizer = Tokenizer()
class Word2VecCBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecCBOW, self).__init__()
        # 输入词嵌入矩阵
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 输出层
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    
    def forward(self, context_words):
        # 获取上下文词的嵌入向量，并计算它们的平均值
        embeds = self.embeddings(context_words)  # (batch_size, context_size, embedding_dim)
        context_embedding = torch.sum(embeds, dim=1)  # (batch_size, embedding_dim)
        
        # 预测中心词的概率分布
        out = self.linear(context_embedding)  # (batch_size, vocab_size)
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, model):
        super().__init__()
        
        self.embedding = model.embeddings
        # print("encoder", model.embeddings.weight.shape)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout,  bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [src_len, batch_size]

        embedded = self.dropout(self.embedding(src))  # [src_len, batch_size, emb_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        
        # outputs: [src_len, batch_size, hid_dim * n_directions]
        # hidden, cell: [n_layers * n_directions, batch_size, hid_dim]
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim] (当前解码器的隐藏状态)
        # encoder_outputs: [src_len, batch_size, hid_dim * 2] (编码器的所有输出)
        
        src_len = encoder_outputs.shape[0]
        
        # Repeat the hidden state for each source word (src_len times)
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # [batch_size, src_len, hid_dim]
        
        # Concatenate hidden state with encoder outputs
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim * 2] 
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention, model):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = model.embeddings
        # print("decoder", model.embeddings.weight.shape)
        self.rnn = nn.LSTM(emb_dim + hid_dim * 2, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 4, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, hidden, cell, encoder_outputs):
        # trg: [batch_size]
        # hidden: [n_layers * n_directions, batch_size, hid_dim]
        # cell: [n_layers * n_directions, batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]

        # print(trg)
        embedded = self.dropout(self.embedding(trg))  # [1, batch_size, emb_dim]
        embedded = embedded.unsqueeze(0)
        # Attention
        
        a = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, src_len, hid_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)  # [1, batch_size, hid_dim * 2]
        # print(embedded.shape, weighted.shape)
        # Concatenate embedding with context vector
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [1, batch_size, emb_dim + hid_dim * 2]
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Prediction
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        # [batch_size, hid_dim * 2] [batch_size, hid_dim * 2] [batch_size, emb_dim]
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # [batch_size, output_dim]
        
        return prediction, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # 第一步输入 <sos> token
        input = trg[0, :]
        output_tokens=[]
        output_log_probs = []
        for t in range(1, 128):

            
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            # outputs[t] = output
            # get the highest predicted token from our predictions

            top1 = output.argmax(1)
            softmax_output = torch.softmax(output, dim=1)
            # print(softmax_output.max(1)[0])
            top1_prob = torch.log(softmax_output.max(1)[0])
            # print(top1_prob)
            output_log_probs.append(top1_prob)

            top1_word = idx2word_en[top1.item()]
            if top1_word == "<EOS>":
                output_tokens.append(top1.item())
                break
            output_tokens.append(top1.item())
            
            # Teacher forcing
            input = trg[t] if random.random() < teacher_forcing_ratio else top1
        perplexity = -torch.stack(output_log_probs).mean()
        return outputs, output_tokens, perplexity
import json
dir_en = "train_data_en.json"
with open(dir_en, "r", encoding='utf-8') as f:
    data_en = json.load(f)
# training_data_en = data_en["training_data"]
word2idx_en = data_en["word2idx"]
dir_jp = "train_data_jp.json"
with open(dir_jp, "r", encoding='utf-8') as f:
    data_jp = json.load(f)
# training_data_jp = data_jp["training_data"]
word2idx_jp = data_jp["word2idx"]
idx2word_en = {idx: word for word, idx in word2idx_en.items()}
idx2word_jp = {idx: word for word, idx in word2idx_jp.items()}
dir_train = "cases.txt"
checkpoint_en = torch.load('en_cbow_model_and_optimizer_20.pth')
cbow_model_en = Word2VecCBOW(len(word2idx_en), 128)
cbow_model_en.load_state_dict(checkpoint_en['model_state_dict'])
# cbow_model_en.eval()
checkpoint_jp = torch.load('jp_cbow_model_and_optimizer_20.pth')
cbow_model_jp = Word2VecCBOW(len(word2idx_jp), 128)
cbow_model_jp.load_state_dict(checkpoint_jp['model_state_dict'])
# cbow_model_jp.eval()

train_data = []
import re
def remove_punctuation_en(word):
    # 匹配所有非字母或数字的字符，并将其替换为空
    return re.sub(r'[^\w\s]', '', word)
def is_punctuation_jp(element):
    # 匹配标点符号
    pattern = r'[、。！？「」『』（）…・]'
    return re.fullmatch(pattern, element) is not None
def make_tokens_jp(sentence):
    tokens =list( tokenizer.tokenize(sentence, wakati=True))
    tokens = [element for element in tokens if not is_punctuation_jp(element)]
    tokens = [word2idx_jp[word] if word in word2idx_jp else word2idx_jp["<UNK>"] for word in tokens]
    return tokens
def make_tokens_en(sentence):
    tokens = []
    for word in sentence:
        word = remove_punctuation_en(word)
        # print(word)
        if word in word2idx_en:
            tokens.append(word2idx_en[word])
        else:
            tokens.append(word2idx_en["<UNK>"])
    return tokens


with open(dir_train, 'r', encoding='utf-8') as file:
    for line in tqdm(file):
        line = line.strip()
        parts = line.split()
        jp_sentence = parts[0]
        en_sentence = parts[1:]
        data = {}
        jp_tokens = make_tokens_jp(jp_sentence)
        en_tokens = make_tokens_en(en_sentence)
        en_tokens.append(word2idx_en["<EOS>"])
        data["jp_tokens"] = jp_tokens
        data["en_tokens"] = en_tokens

        train_data.append(data)
train_data_dir = "cases_data_rnn.json"
with open(train_data_dir, "w", encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(train_data_dir, "r", encoding='utf-8') as f:
    train_data = json.load(f)
jp_sentences = []
en_sentences = []
for data in train_data:
    jp_words = [idx2word_jp[idx] for idx in data["jp_tokens"]]
    en_words = [idx2word_en[idx] for idx in data["en_tokens"]]
    jp_sentences.append(jp_words)
    en_sentences.append(en_words)
vocab_size_en = len(word2idx_en)
vocab_size_jp = len(word2idx_jp)
INPUT_DIM = vocab_size_jp  # 源语言词汇表大小
OUTPUT_DIM = vocab_size_en  # 目标语言词汇表大小
ENC_EMB_DIM = 128  # 编码器嵌入维度
DEC_EMB_DIM = 128  # 解码器嵌入维度
HID_DIM = 512  # LSTM 隐藏维度
N_LAYERS = 2  # LSTM 层数
ENC_DROPOUT = 0.5  # 编码器 Dropout 概率
DEC_DROPOUT = 0.5  # 解码器 Dropout 概率

attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, cbow_model_jp)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn, cbow_model_en)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cbow_model_en = cbow_model_en.to(device)
cbow_model_jp = cbow_model_jp.to(device)

model = Seq2Seq(enc, dec, device).to(device)
checkpoint = torch.load('/root/autodl-tmp/task3/rnn_model_and_optimizer_9.pth')
model.load_state_dict(checkpoint['model_state_dict'])
import torch.optim as optim

# optimizer = optim.SGD(model.parameters(), lr=1e-3)
optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    pbar = tqdm(iterator)
    epoch_loss = 0
    global_step = 0
    total_loss = 0
    for  batch in pbar:
        src = torch.tensor(batch['jp_tokens'], dtype=torch.long)  # 源句子
        trg = torch.tensor(batch['en_tokens'], dtype=torch.long)  # 目标句子

        src = src.to(device)
        trg = trg.to(device)
        src = src.unsqueeze(1)
        trg = trg.unsqueeze(1)

        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # output: [trg_len, batch_size, output_dim]
        output_dim = output.shape[-1]
        
        # 忽略 <sos> token，计算交叉熵损失
        output = output[0:].view(-1, output_dim)
        trg = trg[0:].view(-1)
        
        loss = criterion(output, trg)
        total_loss += loss.item()
        global_step += 1
        if global_step % 100 == 0:

            pbar.set_description(f"loss: {total_loss/100:.4f}")
            
            total_loss = 0
        wandb.log({"loss": loss.item()})
        loss.backward()
        
        # 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        with torch.no_grad():
        
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)
def eval_bleu_perplexity(model, iterator, criterion):
    output_sentences = []
    model.eval()
    epoch_loss = 0
    pbar = tqdm(iterator)
    with torch.no_grad():
        perplexities = []
        for  batch in pbar:
            src = torch.tensor(batch['jp_tokens'], dtype=torch.long)  # 源句子
            trg = torch.tensor(batch['en_tokens'], dtype=torch.long)  # 目标句子

            src = src.to(device)
            trg = trg.to(device)
            src = src.unsqueeze(1)
            trg = trg.unsqueeze(1)

            output, output_tokens, perplexity = model(src, trg, 0)  # turn off teacher
            output_words = [idx2word_en[idx] for idx in output_tokens]
            output_words.insert(0, idx2word_en[trg[0].item()])
            print(output_words)
            output_sentences.append(output_words)
            output_dim = output.shape[-1]
            output = output[0:].view(-1, output_dim)
            trg = trg[0:].view(-1)
            perplexities.append(perplexity.item())
        
        # assert(len(jp_sentences) == len(output_sentences) and len(en_sentences) == len(output_sentences))
        output = []
        perplexity = sum(perplexities) / len(perplexities)
        print("perplexity", perplexity)
        for i in range(len(jp_sentences)):
            item = {}
            item["jp_sentence"] = jp_sentences[i]
            item["en_sentence"] = en_sentences[i]
            item["output_sentence"] = output_sentences[i]
            output.append(item)

        with open("output_cases.json", "w", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)


    return 

N_EPOCHS = 1
CLIP = 1
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_iterator = DataLoader(CustomDataset(train_data), batch_size=1, shuffle=False)

for epoch in range(N_EPOCHS):
    eval_bleu_perplexity(model, train_iterator,  criterion)

