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
import math


#固定随机种子为42
seed=42
random.seed(seed)
np.random.seed(seed)

EasyRec_ReRanker = EasyRec()

    

def relevance_analysis(pos_item_list,target_item,ratio):
    pos_item_profile_list = []
    for pos_item in pos_item_list:
        pos_item_profile = item_feature_to_str(pos_item)
        pos_item_profile_list.append(pos_item_profile)
    embeddings=EasyRec_ReRanker.get_embedding(pos_item_profile_list)
    #转为numpy (11,1024)
    embeddings = np.array(embeddings) 
    target_item_profile = item_feature_to_str(target_item)
    target_embedding=EasyRec_ReRanker.get_embedding([target_item_profile])[0]
    
    selected_indexs=relevance_sampling(embeddings, target_embedding,ratio)
    return selected_indexs




        
parser = argparse.ArgumentParser()
parser.add_argument('--sampling', type=str, default='random', help='choose in random, recent, relevance')
parser.add_argument('--dataset', type=str, default='CDs_and_Vinyl', help='CDs_and_Vinyl')
parser.add_argument('--subset', type=str, default='dense', help='dense')
parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='Backbone LLM model name')
parser.add_argument('--persona_learning_type', type=str, default='pairwise', help='Choose in pairwise, pointwise, or distill. The first two are for reflect, the last one is for distill')
parser.add_argument('--rank_model_name', type=str, default='EasyRec', help='Rank model name')
parser.add_argument('--ratio', type=float, default=0.6, help='ratio')
parser.add_argument('--api_key', type=str, default='', help='OpenAI API key')
args = parser.parse_args()


if args.sampling=='random':
    timestamp = f'{args.persona_learning_type}_{args.sampling}_{args.dataset}_{args.subset}_{args.ratio}'
elif args.sampling=='recent':
    timestamp = f'{args.persona_learning_type}_{args.sampling}_{args.dataset}_{args.subset}_{args.ratio}'
elif args.sampling=='relevance':
    timestamp = f'{args.persona_learning_type}_{args.sampling}_{args.dataset}_{args.subset}_{args.ratio}'
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

#读取jsonl, 获取已经训练好的user_id_list
trained_user_id_list = []
if os.path.exists(f'result/{timestamp}/validation.jsonl'):
    with open(f'result/{timestamp}/validation.jsonl', 'r') as f:
        for line in f:
            trained_user_id_list.append(json.loads(line)['user_id'])
else:
    pass

user_personas = {}
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
        
    
    #开始分析用户行为
    pos_item_list = []
    for item_id in positive_items[:max_index]:
        item_data = data[data['item_id'] == item_id].iloc[0]
        pos_item = {
            key: sanitize_value(value) for key, value in item_data.drop(['rating','timestamp']).to_dict().items()
        }
        pos_item_list.append(pos_item)
    
    #保存到result/{timestamp}/behaviors_count.json
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
        #取天花板数
        sample_size = math.ceil(len(pos_item_list)*args.ratio)
        if args.sampling=='random':
            all_selected_indexs=random.sample(range(len(pos_item_list)), sample_size)
            all_selected_items=[pos_item_list[i] for i in all_selected_indexs]
        elif args.sampling=='recent':
            all_selected_indexs=list(range(len(pos_item_list)))[-sample_size:]
            all_selected_items=[pos_item_list[i] for i in all_selected_indexs]
        elif args.sampling=='relevance':
            target_item_data = user_data.iloc[-1]
            target_item = {
                key: sanitize_value(value) for key, value in target_item_data.drop(['rating','timestamp']).to_dict().items()
            }
            all_selected_indexs=relevance_analysis(pos_item_list,target_item,args.ratio)
            all_selected_items=[pos_item_list[i] for i in all_selected_indexs]
        
        indexed_items = list(zip(all_selected_indexs, all_selected_items))
        indexed_items_sorted = sorted(indexed_items, key=lambda x: x[0])
        sorted_all_selected_items = [item for _, item in indexed_items_sorted]   
    elif args.ratio==0:
        sorted_all_selected_items=[]
    elif args.ratio==1:
        sorted_all_selected_items=pos_item_list
    
    user_selected_items[user_id] = len(sorted_all_selected_items)
    
    for index in range(len(sorted_all_selected_items)):
        pos_item=sorted_all_selected_items[index]
        
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
    
    
    if args.persona_learning_type != 'distill':
        decoupling_prompt = Decoupling_Prompt.format(profile=user_persona)
        response = GPT_QA(decoupling_prompt, model_name=args.model_name, t=0.0, historical_qa=None, siliconflow=False, api_key=args.api_key)
        pos_profile = response.split('Positive Part:')[1].split('Negative Part:')[0].strip()
        neg_profile = response.split('Negative Part:')[1].strip()
        user_personas[user_id] = {'profile':user_persona, 'pos_profile': pos_profile, 'neg_profile': neg_profile}
    else:
        pos_profile=user_persona
        neg_profile='Currently Unknown'
        user_personas[user_id] = {'profile':user_persona, 'pos_profile': pos_profile, 'neg_profile': neg_profile}
    
    
    # 验证,用最后一个数据，加上随机取样的9个负样本，然后调用http://127.0.0.1:8001/rank, 返回的index_list
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
    
    #保存到validation.jsonl
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
    
    #保存user_selected_items.json
    with open(f'result/{timestamp}/user_selected_items.json', 'w') as f:
        json.dump(user_selected_items, f)
    
    

     