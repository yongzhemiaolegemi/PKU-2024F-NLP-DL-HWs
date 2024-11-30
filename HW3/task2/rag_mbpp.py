import json
dir = "mbpp.json"
with open(dir, 'r') as f:
    mbpp_list = json.load(f)
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
code_dir = "code.json"
with open(code_dir, "r") as f:
    code_dataset = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')
code_embeddings = torch.load("code_embeddings.pt")
# print(code_embeddings.shape)
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
mbpp_rag_list = []
for i, item in enumerate(tqdm(mbpp_list)):
    question = item['text']
    embedding = get_embedding(question)
    # print(embedding.shape)
    indices = get_top_k_similar(embedding, code_embeddings, 3)
    # print(indices)
    # for index in indices:
    #     print(code_dataset[index]['instruction'])
    #     print(code_dataset[index]['output'])
    #     print('-----------------')
    # exit()
    question_1 = code_dataset[indices[0]]['instruction']
    answer_1 = code_dataset[indices[0]]['output']
    question_2 = code_dataset[indices[1]]['instruction']
    answer_2 = code_dataset[indices[1]]['output']
    question_3 = code_dataset[indices[2]]['instruction']
    answer_3 = code_dataset[indices[2]]['output']
    item['question_1'] = question_1
    item['answer_1'] = answer_1
    item['question_2'] = question_2
    item['answer_2'] = answer_2
    item['question_3'] = question_3
    item['answer_3'] = answer_3
    mbpp_rag_list.append(item)
print(len(mbpp_rag_list))
output_dir = "mbpp_rag.json"
with open(output_dir, 'w') as f:
    json.dump(mbpp_rag_list, f, indent=4)