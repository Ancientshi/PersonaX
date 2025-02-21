import json
from prompt import Inference_Prompt, Reflect_Prompt, Validate_Prompt, Decoupling_Prompt
import pandas as pd
from utils import GPT_QA, hierarchical_clustering, plot_clustering_heatmap
import random
import logging
import os
from sklearn.model_selection import train_test_split
import time
import argparse
import requests
from utils import ndcg,hit,mrr,sanitize_value,sanitize_data, EasyRec, item_feature_to_str
import numpy as np
import umap 
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from sampling import sampling, relevance_sampling


seed=42
random.seed(seed)
np.random.seed(seed)

EasyRec_ReRanker = EasyRec()
def analysis_behavior_sequence(pos_item_list,user_id,distance_threshold=0.5,alpha=1.1,ratio=0.6):
    if len(pos_item_list)==0:
        print(f'user_id:{user_id} has no item')
        return
    elif len(pos_item_list)==1:
        return [0], pos_item_list, {1: [0]}
    
    pos_item_profile_list = []
    for pos_item in pos_item_list:
        pos_item_profile = item_feature_to_str(pos_item)
        pos_item_profile_list.append(pos_item_profile)
    embeddings=EasyRec_ReRanker.get_embedding(pos_item_profile_list)
    embeddings = np.array(embeddings) 
    
    class2index_list, index2class=hierarchical_clustering(embeddings,  distance_threshold=distance_threshold)
    print('cluster number:',len(class2index_list.keys()))
    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 10: [3, 24, 28], 3: [4, 8, 19], 9: [5], 12: [9], 11: [10, 14, 31], 2: [11, 17, 21], 6: [12, 27], 4: [13, 32], 8: [15, 29], 5: [20], 7: [22, 23]}
    
    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 12: [3, 24, 28], 4: [4], 11: [5], 3: [8, 19], 14: [9], 13: [10, 14, 31], 2: [11, 17, 21], 7: [12], 5: [13, 32], 10: [15, 29], 6: [20], 9: [22, 23], 8: [27]}
    
    # {1: [0, 1, 2, 6, 7, 16, 18, 25, 26, 30, 33], 4: [3, 10, 14, 24, 28, 31], 2: [4, 8, 11, 17, 19, 21], 3: [5, 12, 13, 15, 20, 22, 23, 27, 29, 32], 5: [9]}
    
    # print(class2index_list)
    # aa=input('pause')
    
    cluster_list=[]
    class2centroid={}
    for class_id, index_list in class2index_list.items():
        points = embeddings[index_list]
        cluster_list.append(points)
        centroid = np.mean(points, axis=0)
        class2centroid[class_id]=centroid
    
    selected_indexs_list=sampling(cluster_list, alpha=alpha, ratio=ratio)  
    selected_class2index_list = {}
    for (class_label, index_list), selected_indexs in zip(class2index_list.items(), selected_indexs_list):
        selected_class2index_list[class_label] = [index_list[i] for i in selected_indexs]

    all_selected_indexs = []
    for index_list in selected_class2index_list.values():
        all_selected_indexs += index_list
     
        
    all_selected_items=[]
    for index in all_selected_indexs:
        all_selected_items.append(pos_item_list[index])
    
    return all_selected_indexs, all_selected_items, selected_class2index_list,class2centroid
        

def relevance_analysis(centroid_list,target_item):
    embeddings = np.array(centroid_list)
    target_item_profile = item_feature_to_str(target_item)
    target_embedding=EasyRec_ReRanker.get_embedding([target_item_profile])[0]
    relevance_scores=np.dot(embeddings,target_embedding)
    selected_index=np.argsort(relevance_scores)[-1]
    return selected_index

        
parser = argparse.ArgumentParser()
parser.add_argument('--sampling', type=str, default='CBS_relevance', help='choose in CBS_relevance')
parser.add_argument('--dataset', type=str, default='CDs_and_Vinyl', help='CDs_and_Vinyl')
parser.add_argument('--subset', type=str, default='dense', help='dense')
parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Backbone LLM model name')
parser.add_argument('--persona_learning_type', type=str, default='pairwise', help='Choose in pairwise, pointwise, or distill. The first two are for reflect, the last one is for distill')
parser.add_argument('--rank_model_name', type=str, default='EasyRec', help='Rank model name')
parser.add_argument('--distance_threshold', type=float, default=0.5, help='distance_threshold')
parser.add_argument('--alpha', type=float, default=1.06, help='alpha')
parser.add_argument('--ratio', type=float, default=0.6, help='ratio')
parser.add_argument('--api_key', type=str, default='', help='OpenAI API key')
args = parser.parse_args()

if args.sampling=='CBS_relevance':
    timestamp = f'{args.persona_learning_type}_{args.sampling}_{args.dataset}_{args.subset}_{args.distance_threshold}_{args.alpha}_{args.ratio}'
else:
    raise ValueError('sampling should be in CBS, random, recent, relevance, CBS_relevance')

    
# Create log directory if it doesn't exist
if not os.path.exists(f'result/{timestamp}'):
    os.makedirs(f'result/{timestamp}')

# Custom verbose level
VERBOSE_LEVEL = 25
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

# Define the verbose method
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(VERBOSE_LEVEL):
        self._log(VERBOSE_LEVEL, message, args, **kwargs)

# Add verbose method to logger
logging.Logger.verbose = verbose

# Configure logging
logging.basicConfig(level=VERBOSE_LEVEL, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
                    datefmt='%Y-%m-%d %H:%M:%S %Z',
                    filename=f'result/{timestamp}/default.log',
                    filemode='a')

# Get logger
logger = logging.getLogger(__name__)

def uniform_sampling(candidate_items):
    return random.choice(list(candidate_items))


dataset_path = f'Amazon/{args.dataset}/sampled_{args.subset}.csv'
data = pd.read_csv(dataset_path)

unique_users = data['user_id'].unique().tolist()
unique_items = data['item_id'].unique().tolist()

#Read the jsonl and get the trained user_id_list
trained_user_id_list = []
if os.path.exists(f'result/{timestamp}/validation.jsonl'):
    with open(f'result/{timestamp}/validation.jsonl', 'r') as f:
        for line in f:
            trained_user_id_list.append(json.loads(line)['user_id'])
else:
    pass

user_personas = {}
ndcg_dict = {}
user_selected_items = {}

for user_id in unique_users:
    if user_id in trained_user_id_list:
        continue

    learning_request_data={}
    user_persona = user_personas.get(user_id, 'Currently Unknown')
    learning_request_data['type'] = args.persona_learning_type
    learning_request_data['model_name'] = args.model_name
    learning_request_data['api_key'] = args.api_key
    learning_request_data['user'] = {'user_id':user_id,'user_persona': user_persona}
    learning_request_data['sequence'] = []
    
    
    user_data = data[data['user_id'] == user_id].sort_values(by='timestamp')
    user_data = user_data[user_data['rating'] >= 1]
    
    positive_items = user_data['item_id'].tolist()
    negative_candidates = set(unique_items) - set(positive_items)
    
    
    if len(user_data)==0:
        print(f'user_id:{user_id} has no item')
        continue
    elif len(user_data)==1:
        max_index=1
        print(f'user_id:{user_id} has only one item')
    else:
        max_index=len(user_data)-1
        
    
    pos_item_list = []
    for item_id in positive_items[:max_index]:
        item_data = data[data['item_id'] == item_id].iloc[0]
        pos_item = {
            key: sanitize_value(value) for key, value in item_data.drop(['rating','timestamp']).to_dict().items()
        }
        pos_item_list.append(pos_item)
    
    if os.path.exists(f'result/{timestamp}/behaviors_count.json'):
        with open(f'result/{timestamp}/behaviors_count.json', 'r') as f:
            existed_behaviors_count_dict = json.load(f)
        existed_behaviors_count_dict[user_id] = len(pos_item_list)
        with open(f'result/{timestamp}/behaviors_count.json', 'w') as f:
            json.dump(existed_behaviors_count_dict, f)
    else:
        with open(f'result/{timestamp}/behaviors_count.json', 'w') as f:
            json.dump({user_id: len(pos_item_list)},f)
    

    if 0<args.ratio<1:
        all_selected_indexs, all_selected_items, selected_class2index_list,class2centroid=analysis_behavior_sequence(pos_item_list,user_id,args.distance_threshold,args.alpha,args.ratio)
        
        print('len(all_selected_items)',len(all_selected_items))
        

        target_item_data = user_data.iloc[-1]
        target_item = {
            key: sanitize_value(value) for key, value in target_item_data.drop(['rating','timestamp']).to_dict().items()
        }
        selected_index=relevance_analysis(list(class2centroid.values()),target_item)
        class_label = list(class2centroid.keys())[selected_index]
        index_list=selected_class2index_list[class_label]
        
        #Once you have selected the appropriate centroid, proceed with persona_learning
        print('class_label:',class_label,'len(index_list):',len(index_list))
        user_selected_items[user_id] = len(index_list)
        

        learning_request_data['sequence']=[]
        for index in index_list:
            pos_item=pos_item_list[index]
            
            
            negative_item_id = uniform_sampling(negative_candidates)
            negative_item_row = data[data['item_id'] == negative_item_id].iloc[0].drop(['rating','timestamp'])
            neg_item = {
                key: sanitize_value(value) for key, value in negative_item_row.to_dict().items()
            }
            learning_request_data['sequence'].append({'pos_item': pos_item, 'neg_item': neg_item})
            
        
        #post调用 127.0.0.1:8001/persona_learning
        try:
            learning_response = requests.post('http://127.0.0.1:8001/persona_learning', json=learning_request_data).json()
        except Exception as e:
            print(learning_request_data)
            logger.error(f'Error: {e}')
            aa=input('error happened, pause')
            
        user_persona=learning_response['user_persona']
        log=learning_response['log']

        verbose_dict={'api':'persona_learning','request_data':learning_request_data,'response':learning_response}
        logger.verbose(json.dumps(verbose_dict, indent=4))    
        ###
        
        pos_profile=user_persona
        neg_profile='Currently Unknown'
        user_personas[user_id] = {'profile':user_persona, 'pos_profile': pos_profile, 'neg_profile': neg_profile}
    elif args.ratio==0:
        sorted_all_selected_items=[]
    elif args.ratio==1:
        sorted_all_selected_items=pos_item_list
    

    # Validation, with the final data, combined with random sampling of nine negative samples, and then call http://127.0.0.1:8001/rank, return index_list
    rank_request_data={}
    rank_request_data['user']={'user_id':user_id,'user_persona': user_persona,'pos_user_persona': pos_profile,'neg_user_persona': neg_profile}
    rank_request_data['items'] = [sanitize_data(user_data.iloc[-1].drop(['rating', 'timestamp']).to_dict())] + [sanitize_data(data[data['item_id'] == i].iloc[0].drop(['rating', 'timestamp']).to_dict()) for i in random.sample(list(negative_candidates), 9)]
    rank_request_data['model_name'] = args.rank_model_name
    rank_request_data['api_key'] = args.api_key
    
    
    
    try:
        rank_response = requests.post('http://127.0.0.1:8001/rank_local', json=rank_request_data).json()
    except Exception as e:
        logger.error(f'Error: {e}')
        print(rank_request_data)
        aa=input('error happened, pause')
    
    index_list=rank_response['index_list']
    verbose_dict={'api':'rank','request_data':rank_request_data,'response':rank_response}
    logger.verbose(json.dumps(verbose_dict, indent=4))
    

    test_truth_list = [[0]]
    test_prediction_list = [index_list]
    topk = [1,5,10]
    ndcgs = ndcg(test_truth_list, test_prediction_list, topk)
    hits = hit(test_truth_list, test_prediction_list, topk)
    mrrs = mrr(test_truth_list, test_prediction_list, topk)
    metrics = {'ndcg@1': ndcgs[0], 'ndcg@5': ndcgs[1], 'ndcg@10': ndcgs[2], 'hit@1': hits[0], 'hit@5': hits[1], 'hit@10': hits[2], 'mrr@1': mrrs[0], 'mrr@5': mrrs[1], 'mrr@10': mrrs[2]}
    
    print(metrics)
    
    with open(f'result/{timestamp}/validation.jsonl', 'a') as f:
        f.write(json.dumps({'user_id': user_id, 'metrics': metrics})+'\n')
    
    if os.path.exists(f'result/{timestamp}/user_personas.json'):
        with open(f'result/{timestamp}/user_personas.json', 'r') as f:
            existed_user_personas_dict = json.load(f)
        existed_user_personas_dict.update(user_personas)
        with open(f'result/{timestamp}/user_personas.json', 'w') as f:
            json.dump(existed_user_personas_dict, f)
    else:
        with open(f'result/{timestamp}/user_personas.json', 'w') as f:
            json.dump(user_personas, f)
    
    with open(f'result/{timestamp}/user_selected_items.json', 'w') as f:
        json.dump(user_selected_items, f)
    
    

     