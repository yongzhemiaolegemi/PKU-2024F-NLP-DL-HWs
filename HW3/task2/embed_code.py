from sentence_transformers import SentenceTransformer
import torch
# 加载预训练的模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 需要获取 embedding 的句子

import json
code_dir = "code.json"
with open(code_dir, "r") as f:
    code_dataset = json.load(f)
# code_dataset = code_dataset[:100]
from tqdm import tqdm
code_embeddings = None
for i, item in enumerate(tqdm(code_dataset)):
    instruction = item["instruction"]
    input = item["input"]
    output = item["output"] 
    full_text = "# Instruction: " + instruction + "\n" + "# Input: "+input + "\n" +"# Output: "+ output
# 获取句子 embedding
    embedding = model.encode(full_text)
    embedding = torch.tensor(embedding).unsqueeze(0)
    if code_embeddings is None:
        code_embeddings = embedding
    else:
        code_embeddings = torch.cat((code_embeddings, embedding), 0)

print(code_embeddings.shape)
# save tensor
torch.save(code_embeddings, "code_embeddings.pt")
