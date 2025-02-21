import json
import requests
from flask import Flask, render_template, request, jsonify, Response, session
import os
from core import distill, train, rank_local


# Configuration and constants
SECRET_KEY = os.urandom(24).hex()
app = Flask(__name__)
app.secret_key = SECRET_KEY

@app.route('/persona_learning', methods=['POST'])
def personal_learning():
    data = request.json
    type = data['type'] #pointwise or pairwise for Reflection, distill for Summarization
    user = data['user']
    sequence = data['sequence'] #each pair should have pos_item_id and neg_item_id
    
    #get model_name, siliconflow, api_key
    model_name = data.get('model_name', None)
    api_key = data.get('api_key', None)
    
    if model_name is None:
        return jsonify({'error': 'model_name is required','status':400})
    else:
        if model_name in ['gpt-4o','gpt-4o-mini']:
            siliconflow = False
        else:
            siliconflow = True
    
    if api_key is None:
        return jsonify({'error': 'api_key is required','status':400})
    
    if type == 'distill':
        user_persona,log=distill(user, sequence,type,model_name,siliconflow,api_key)
    else:
        user_persona,log=train(user, sequence,type,model_name,siliconflow,api_key)
    return jsonify({'user_persona': user_persona,'log':log,'status':200})


@app.route('/rank_local', methods=['POST'])
def preference_rank_local():
    data = request.json
    user = data['user']
    items = data['items']
    model_name = data.get('model_name', None)
    api_key = data.get('api_key', None)
    
    siliconflow = False
    assert model_name in ['BGE','EasyRec']
    
    if api_key is None:
        return jsonify({'error': 'api_key is required','status':400})
    
    index_list,relevance_score_list = rank_local(user, items, model_name, siliconflow, api_key)
    return jsonify({'index_list': index_list,'relevance_score_list':relevance_score_list,'status':200})



# Run the Flask application
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8001, debug=False)
