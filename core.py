from utils import GPT_QA,item_feature_to_str
from prompt import Inference_Prompt, Validate_Prompt, Distillation_Prompt, Reflect_Prompt, Decoupling_Prompt
from rerank import *
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import random
import torch


#seed=42固定
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

        

def reflect(user_profile, negative_item_profile, positive_item_profile, response,model_name,siliconflow,api_key):
    reflect_prompt = Reflect_Prompt.format(profile=user_profile, item_a=negative_item_profile, item_b=positive_item_profile, response=response)
    response = GPT_QA(reflect_prompt, model_name=model_name, t=0.0, historical_qa=None, siliconflow=siliconflow, api_key=api_key)
    updated_user_profile = None
    for line in response.split('\n'):
        if 'My updated profile:' in line:
            updated_user_profile = line.replace('My updated profile:', '').strip()
    if updated_user_profile is None:
        raise Exception(f'User Profile Reflect Error: {response}')
    else:
        return updated_user_profile
    



def distill(user, sequence, learning_type,model_name,siliconflow,api_key):
    log = {}
    
    user_profile = user.get('user_persona', 'Currently Unknown')
    
    pos_item_profile_list = []
    for pair in sequence:
        pos_item = pair['pos_item']
        pos_item_profile = item_feature_to_str(pos_item)
        pos_item_profile_list.append(pos_item_profile)
    
    sequence_item_profile=''''''
    for i,pos_item_profile in enumerate(pos_item_profile_list):
        sequence_item_profile+=f"Item {i}. {pos_item_profile}\n"
    
    distillation_prompt = Distillation_Prompt.format(profile=user_profile, sequence_item_profile=sequence_item_profile)
    response = GPT_QA(distillation_prompt, model_name=model_name, t=0.0, historical_qa=None, siliconflow=siliconflow, api_key=api_key)
    
    try:
        user_profile=response.split('Summarization:')[-1].strip()
    except:
        print(response)
        raise ValueError(f"Error in LLM response, cannot summarize user profile. Response: {response}")
    
    log['user_profile']=user_profile
    log['sequence_item_profile']=sequence_item_profile
    log['response']=response

    return user_profile, log

def train(user, sequence, learning_type,model_name,siliconflow,api_key):
    """
    :param user: The information of the user.
    :param sequence: A sequence of pairs where each pair contains pos_item_id and neg_item_id.
    :param learning_type: The type of learning (pointwise or pairwise).
    :return: The updated user persona and logs.
    """
    log = []
    
    user_profile = user.get('user_persona', 'Currently Unknown')

    max_try=3
    pair_index=0
    try_times=0
    while pair_index<len(sequence):
        pair=sequence[pair_index]
        pos_item = pair['pos_item']
        neg_item = pair['neg_item']
        
        pos_item_profile = item_feature_to_str(pos_item)
        neg_item_profile = item_feature_to_str(neg_item)

        if learning_type == 'pairwise':
            inference_prompt = Inference_Prompt.format(profile=user_profile, item_a=neg_item_profile, item_b=pos_item_profile)
        else:
            inference_prompt = Validate_Prompt.format(profile=user_profile, item=pos_item_profile)
        
        response = GPT_QA(inference_prompt, model_name=model_name, t=0.0, historical_qa=None, siliconflow=siliconflow, api_key=api_key)

        choose_item = None
        explanation = None
        
        try:
            choose_item=response.split('Chosen Item:')[1].split('Explanation:')[0].strip()
            explanation=response.split('Explanation:')[1].strip()
        except:
            print(response)
            raise ValueError(f"Error in LLM response, cannot find Chosen Item or Explanation. Response: {response}")

        item_a_title = neg_item['title']
        item_b_title = pos_item['title']
            
        # Update user profile based on GPT's answers
        if ('Item A' in choose_item and 'Item B' in choose_item) or (item_a_title in choose_item and item_b_title in choose_item):
            raise ValueError(f"Error in LLM response, both Item A and Item B chosen. Response: {response}")
        elif 'Item A' in choose_item or item_a_title in choose_item or user_profile=='Currently Unknown':  # The negative sample is selected, indicating that the reasoning is wrong and needs to be updated; Or the user_profile initialization is Currently Unknown, forcing the update
            try:
                user_profile = reflect(user_profile=user_profile, 
                                       negative_item_profile=neg_item_profile, 
                                       positive_item_profile=pos_item_profile, 
                                       response=response,model_name=model_name,siliconflow=siliconflow,api_key=api_key)

                log_entry = {
                    'user': user,
                    'pair': pair,
                    'response': response,
                    'reflection': {
                        'user_profile': user_profile,
                    },
                }
                log.append(log_entry)
                
                if try_times<max_try:
                    try_times+=1
                else: 
                    pair_index+=1
                    try_times=0
        
            except Exception as e:
                raise Exception(f"Error updating profile for user {user}: {str(e)}")
        elif 'Item B' in choose_item or item_b_title in choose_item:  # Positive sample selected, correct reasoning
            # Do not need to update the user portrait when correct, continue to the next sample
            log_entry = {
                'user': user,
                'pair': pair,
                'response': response,
                'reflection': None,
            }
            log.append(log_entry)
            
            pair_index+=1
            try_times=0
        else:
            raise ValueError(f"Error in LLM response, neither Item A nor Item B chosen. Response: {response}")
    
    return user_profile, log


def compute_score_parallel(profile, item_profile_list, method):
    """
    :param profile: Personal information about the user or item
    :param item_profile_list: A list of features of the item
    :return: Index list and relevance score list
    """
    return compute_score(profile, item_profile_list,method)

def rank_local(user, items, model_name, siliconflow, api_key):
    """
    :param user: The information of the user.
    :param items: A list of items to rank.
    :return: The ranked list of items.
    """
    ranked_items = []
    user_profile = user.get('user_persona', 'Currently Unknown')
    pos_profile = user.get('pos_user_persona', 'Currently Unknown')
    neg_profile = user.get('neg_user_persona', 'Currently Unknown')

    item_profile_list = []
    for item in items:
        item_profile = item_feature_to_str(item)
        item_profile_list.append(item_profile)

    if pos_profile != 'Currently Unknown' and neg_profile != 'Currently Unknown':
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_pos = executor.submit(compute_score_parallel, pos_profile, item_profile_list, model_name)
            future_neg = executor.submit(compute_score_parallel, neg_profile, item_profile_list, model_name)
            
            pos_index_list, pos_relevance_score_list = future_pos.result()
            neg_index_list, neg_relevance_score_list = future_neg.result()

        #The higher the pos_relevance_score_list, the more it is recommended, and the higher the neg_relevance_score_list, the more it cannot be recommended
        relevance_score_list = np.array(pos_relevance_score_list) - np.array(neg_relevance_score_list)
    elif pos_profile != 'Currently Unknown' and neg_profile == 'Currently Unknown':
        index_list, relevance_score_list = compute_score(pos_profile, item_profile_list, model_name)
        relevance_score_list = np.array(relevance_score_list)
    elif pos_profile == 'Currently Unknown' and neg_profile == 'Currently Unknown':
        index_list, relevance_score_list = compute_score(user_profile, item_profile_list, model_name)
        relevance_score_list = np.array(relevance_score_list)
        
    # Rank from most to least based on relevance_score
    index_list = np.argsort(relevance_score_list)[::-1].tolist()

    return index_list, relevance_score_list.tolist()
