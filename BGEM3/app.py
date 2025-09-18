from flask import Flask, request, jsonify
from main import get_embeddings
import torch
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

# Initialize Flask app
app = Flask(__name__)

#get_embedding
@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    data = request.get_json()
    
    # Get the query and documents from the request
    documents = data.get('documents')

    embeddings = get_embeddings(documents)
    
    return jsonify({"embeddings": embeddings.tolist()}) # Convert embeddings to list and return
    
if __name__ == '__main__':
    # Run the Flask app on port 8500
    app.run(host='0.0.0.0', port=8502)
