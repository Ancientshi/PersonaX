import json
import openai
import requests
import asyncio
import os
import sys
import requests
import re
import string
import numpy as np
from langchain_community.document_loaders import AsyncHtmlLoader,BSHTMLLoader
from langchain_community.document_transformers import Html2TextTransformer
import concurrent.futures
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


cache_dir = "./model"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

os.environ['OPENAI_API_KEY'] = ''
os.environ['SILICONFLOW_API_KEY'] = ''

def GPT_QA(prompt, model_name="gpt-3.5-turbo-16k", t=0.0,historical_qa=None,siliconflow=False,api_key=None):
    if siliconflow:
        url = "https://api.siliconflow.cn/v1/chat/completions"
        if api_key is not None:
            openai.api_key =api_key
        else:
            openai.api_key =os.environ["SILICONFLOW_API_KEY"]
    else:
        url = "https://api.openai.com/v1/chat/completions"
        if api_key is not None:
            openai.api_key =api_key
        else:
            openai.api_key =os.environ["OPENAI_API_KEY"]
        
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai.api_key}"}
    messages=[]
    if historical_qa!=None:
        for (q,a) in historical_qa:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt})
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": t,
        "n": 1,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
    except:
        print("Error: Connection error")
        answer="Connection error"
    try: 
        answer = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        answer=f"Connection error, {response.json()}"
    return answer


    
    



def ndcg(test_truth_list, test_prediction_list, topk):
    ndcgs = []
    
    for k in topk:
        ndcg_list = []
        
        for ind, test_truth in enumerate(test_truth_list):
            dcg = 0
            idcg = 0
            test_truth_index = set(test_truth)
            
            if len(test_truth_index) == 0:
                continue
            
            top_sorted_index = test_prediction_list[ind][0:k]
            
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    dcg += 1.0 / np.log2(index + 2)  # index从0开始，因此 +2

            sorted_truth_index = list(test_truth)[:k]
            for ideal_index in range(min(len(sorted_truth_index), k)):
                idcg += 1.0 / np.log2(ideal_index + 2)

            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
                
            ndcg_list.append(ndcg)
        
        if len(ndcg_list) > 0:
            ndcgs.append(np.mean(ndcg_list))
        else:
            ndcgs.append(0.0)
    
    return ndcgs

def hit(test_truth_list, test_prediction_list, topk):
    hits = []
    
    for k in topk:
        hit_list = []
        
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_set = set(test_truth)
            
            if len(test_truth_set) == 0:
                continue
            
            top_sorted_index = test_prediction_list[ind][0:k]
            
            hit = 1.0 if any(item in test_truth_set for item in top_sorted_index) else 0.0
            hit_list.append(hit)
        
        if len(hit_list) > 0:
            hits.append(np.mean(hit_list))
        else:
            hits.append(0.0)
    
    return hits


def mrr(test_truth_list, test_prediction_list, topk):
    mrrs = []
    
    for k in topk:
        mrr_list = []
        
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_set = set(test_truth)
            
            if len(test_truth_set) == 0:
                continue
            
            top_sorted_index = test_prediction_list[ind][0:k]
            
            mrr = 0.0
            for index, item in enumerate(top_sorted_index):
                if item in test_truth_set:
                    mrr = 1.0 / (index + 1)  
                    break
            
            mrr_list.append(mrr)
        
        if len(mrr_list) > 0:
            mrrs.append(np.mean(mrr_list))
        else:
            mrrs.append(0.0)
    
    return mrrs


def sanitize_value(value):
    if isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')):
        return None
    return value

def sanitize_data(data):
    return {
        key: sanitize_value(value) for key, value in data.items()
    }
    



class EasyRec:
    def __init__(self, url='http://localhost:8500'):
        self.url = url

    def get_embedding(self, documents):
        response = requests.post(f"{self.url}/get_embedding", json={"documents": documents})
        embeddings = response.json()['embeddings']
        return embeddings
    
    def predict(self, query, documents):
        response = requests.post(f"{self.url}/compute_scores", json={"query": query, "documents": documents})
        scores = response.json()['scores']
        return scores
    
    



import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import os
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
from sklearn.utils import shuffle
from collections import defaultdict

def hierarchical_clustering(embeddings, distance_threshold=0.3):
    Z = linkage(embeddings, method='ward', metric='euclidean')
    
    labels = fcluster(Z, t=distance_threshold, criterion='distance')
    
    original_indices = np.arange(len(embeddings))
    
    class2index_list = defaultdict(list)
    index2class = {}
    
    for original_idx, label in zip(original_indices, labels):
        class2index_list[label].append(original_idx)
        index2class[original_idx] = label
    
    return dict(class2index_list), index2class

def plot_clustering_heatmap(embeddings, class2index_list, distance_threshold=0.3, user_id='user', timestamp='timestamp'):
    distance_matrix = squareform(pdist(embeddings, metric='euclidean'))
    
    Z = linkage(embeddings, method='ward', metric='euclidean')

    dendro = dendrogram(Z, no_plot=True)
    ordered_indices = dendro['leaves'] 
    
    original_indices = np.arange(len(embeddings))
    
    ordered_original_indices = original_indices[ordered_indices]

    ordered_distance_matrix = distance_matrix[np.ix_(ordered_indices, ordered_indices)]


    color_map = plt.get_cmap('Set3', len(class2index_list))
    color_dict = {key: color_map(key%20) for key in class2index_list.keys()}

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        ordered_distance_matrix,
        annot=False,
        cmap='viridis',
        xticklabels=ordered_original_indices,
        yticklabels=ordered_original_indices,
        cbar_kws={'label': 'Distance'},
        linewidths=0.5,
        linecolor='black',
    )

    ax = plt.gca()
    
    
    index2class = {}
    for key, indices in class2index_list.items():
        for index in indices:
            index2class[index] = key
        
            
    for label in ax.get_xticklabels():
        index = int(label.get_text())
        class_value=index2class[index]
        label.set_color(color_dict[class_value])  
        
    for label in ax.get_yticklabels():
        index = int(label.get_text())
        class_value=index2class[index]
        label.set_color(color_dict[class_value])   

    plt.title('Euclidean Distance Heatmap (Ordered by Clusters)')

    os.makedirs(f'figure/{timestamp}', exist_ok=True)
    plt.savefig(f'figure/{timestamp}/heatmap_{user_id}.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=ordered_original_indices.astype(str), leaf_rotation=90, leaf_font_size=10)
    plt.title('Hierarchical Clustering Dendrogram')

    plt.savefig(f'figure/{timestamp}/dendrogram_{user_id}.png')
    plt.clf()

    print(f'Figures saved to figure/{timestamp}/')
    
    
def item_feature_to_str(item_feature):
    """
    Convert item feature to string. 
    :param item_feature: The feature of an item, which is a dictionary.
    """
    assert isinstance(item_feature, dict), f"item_feature should be a dictionary, but got {type(item_feature)}"
    feature_str = ""
    for key,value in item_feature.items():
        if 'id' in key:
            continue
        else:
            feature_str += f"{key}:{value}\n"
    return feature_str