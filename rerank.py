from FlagEmbedding import FlagReranker
import torch
import numpy as np
import random
import requests


#seed=42固定
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device='cuda:0'
BGE_Reranker = FlagReranker('', use_fp16=True, device=device)

def compute_score(query, doc_list, method='BGE'):
    if method == 'BGE':
        # 计算分数
        scores = BGE_Reranker.compute_score([[query, doc] for doc in doc_list], normalize=True)
    elif method=='EasyRec':
        #访问本地8500端口，/compute_scores, scores是一个list
        response = requests.post('http://localhost:8500/compute_scores', json={'query': query, 'documents': doc_list}).json()
        scores = response['scores']
        
        
    
    # 获取每个文档的索引及其对应的分数
    indexed_scores = list(enumerate(scores))
    
    # 根据分数对索引进行排序
    sorted_indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    
    # 提取排序后的索引
    sorted_indices = [index for index, score in sorted_indexed_scores]
    
    return sorted_indices, scores

