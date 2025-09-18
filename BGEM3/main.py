import os
from FlagEmbedding import BGEM3FlagModel
import torch
import numpy as np
import random


BGEM3_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device="cuda:0")

# 设置缓存目录
os.environ["TRANSFORMERS_CACHE"] = './model'

def get_embeddings(doc_list):
    embeddings = BGEM3_model.encode(doc_list, 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
    return embeddings