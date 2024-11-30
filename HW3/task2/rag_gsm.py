import json
dir = "gsm8k.json"
with open(dir, 'r') as f:
    gsm_list = json.load(f)
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
math_dir = "math.json"
with open(math_dir, "r") as f:
    math_dataset = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
math_embeddings = torch.load("math_embeddings.pt")
# print(math_embeddings.shape)
def get_top_k_similar(embedding, embeddings, k):
    query = embedding.numpy()
    embedding_matrix = embeddings.numpy()
    similarities = cosine_similarity(query, embedding_matrix)

    # 找到与query最相似的3个embedding的索引
    top_k_indices = similarities.argsort()[0][-3:][::-1]

    return top_k_indices
def get_embedding(question):

    embedding = model.encode(question)
    embedding = torch.tensor(embedding).unsqueeze(0)
    return embedding
gsm_rag_list = []
for i, item in enumerate(tqdm(gsm_list)):
    question = item['question']
    embedding = get_embedding(question)
    # print(embedding.shape)
    indices = get_top_k_similar(embedding, math_embeddings, 3)
    # print(indices)
    # for index in indices:
    #     print(math_dataset[index]['problem'])
    #     print(math_dataset[index]['solution'])
    #     print('-----------------')
    question_1 = math_dataset[indices[0]]['problem']
    answer_1 = math_dataset[indices[0]]['solution']
    question_2 = math_dataset[indices[1]]['problem']
    answer_2 = math_dataset[indices[1]]['solution']
    question_3 = math_dataset[indices[2]]['problem']
    answer_3 = math_dataset[indices[2]]['solution']
    item['question_1'] = question_1
    item['answer_1'] = answer_1
    item['question_2'] = question_2
    item['answer_2'] = answer_2
    item['question_3'] = question_3
    item['answer_3'] = answer_3
    gsm_rag_list.append(item)
print(len(gsm_rag_list))
output_dir = "gsm_rag.json"
with open(output_dir, 'w') as f:
    json.dump(gsm_rag_list, f, indent=4)