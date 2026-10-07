"""HTTP clients, item formatting, and behavior clustering for PersonaX."""

import os
from collections import defaultdict

import numpy as np
import requests
from scipy.cluster.hierarchy import linkage, fcluster


def GPT_QA(prompt, model_name="gpt-3.5-turbo-16k", t=0.0,historical_qa=None,siliconflow=False,api_key=None):
    if siliconflow:
        url = "https://api.siliconflow.cn/v1/chat/completions"
        if api_key is not None:
            request_api_key = api_key
        else:
            request_api_key = os.environ["SILICONFLOW_API_KEY"]
    else:
        url = "https://api.openai.com/v1/chat/completions"
        if api_key is not None:
            request_api_key = api_key
        else:
            request_api_key = os.environ["OPENAI_API_KEY"]

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {request_api_key}"}
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
