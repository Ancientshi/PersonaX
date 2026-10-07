import numpy as np
import random
import requests


#seed=42固定
seed=42
random.seed(seed)
np.random.seed(seed)


def compute_score(query, doc_list, method='EasyRec'):
    if method=='EasyRec':
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

