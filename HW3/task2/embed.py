from sentence_transformers import SentenceTransformer
import torch
# 加载预训练的模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 需要获取 embedding 的句子

import json
math_dir = "math.json"
with open(math_dir, "r") as f:
    math_dataset = json.load(f)
# math_dataset = math_dataset[:100]
from tqdm import tqdm
math_embeddings = None
for i, item in enumerate(tqdm(math_dataset)):
    problem = item["problem"]
    solution = item["solution"]
    type = item["type"]
    full_text = "# Type: " + type + "\n" + "# Problem: "+problem + "\n" +"#Solution: "+ solution
# 获取句子 embedding
    embedding = model.encode(full_text)
    embedding = torch.tensor(embedding).unsqueeze(0)
    if math_embeddings is None:
        math_embeddings = embedding
    else:
        math_embeddings = torch.cat((math_embeddings, embedding), 0)

print(math_embeddings.shape)
# save tensor
torch.save(math_embeddings, "math_embeddings.pt")
