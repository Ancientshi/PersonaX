from flask import Flask, request, jsonify
import torch
from model import Easyrec
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import random


#设置seed
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#seed=42固定
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Set custom cache directory
cache_dir = "/home/yunxshi/Data/workspace/EasyRec/model"  # Change this path to your desired cache location
# Check if CUDA is available and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load the config, model, and tokenizer with the custom cache directory
config = AutoConfig.from_pretrained("hkuds/easyrec-roberta-large", cache_dir=cache_dir)
# model = Easyrec.from_pretrained("hkuds/easyrec-roberta-large", config=config, cache_dir=cache_dir,  quantization_config=nf4_config)
model = Easyrec.from_pretrained("hkuds/easyrec-roberta-large", config=config, cache_dir=cache_dir)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("hkuds/easyrec-roberta-large", use_fast=True, cache_dir=cache_dir)



# Initialize Flask app
app = Flask(__name__)

@app.route('/compute_scores', methods=['POST'])
def compute_scores():
    data = request.get_json()
    
    # Get the query and documents from the request
    query = data.get('query')
    documents = data.get('documents')

    if not query or not isinstance(query, str):
        return jsonify({"error": "Invalid or missing 'query'"}), 400
    if not documents or not isinstance(documents, list):
        return jsonify({"error": "Invalid or missing 'documents'"}), 400

    # Combine query and documents into profiles list
    profiles = [query] + documents

    # Tokenize the input profiles
    inputs = tokenizer(profiles, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)  # Move inputs to GPU
    
    # Compute embeddings using the model
    with torch.inference_mode():
        embeddings = model.encode(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings.pooler_output.detach().float(), dim=-1)

    # Calculate scores between query and each document
    query_embedding = embeddings[0]
    score_list = []
    for i in range(1, len(profiles)):
        score = torch.matmul(query_embedding, embeddings[i]).item()
        score_list.append(score)

    return jsonify({"scores": score_list})



#get_embedding
@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    data = request.get_json()
    
    # Get the query and documents from the request
    documents = data.get('documents')

    if not documents or not isinstance(documents, list):
        return jsonify({"error": "Invalid or missing 'documents'"}), 400
    
    # Tokenize the input profiles
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    # Compute embeddings using the model
    with torch.inference_mode():
        embeddings = model.encode(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings.pooler_output.detach().float(), dim=-1)
    
    return jsonify({"embeddings": embeddings.tolist()}) # Convert embeddings to list and return
    
if __name__ == '__main__':
    # Run the Flask app on port 8500
    app.run(host='0.0.0.0', port=8500)
