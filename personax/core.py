"""Persona distillation and reflection from interaction histories."""

import random

import numpy as np

from .prompt import Inference_Prompt, Validate_Prompt, Distillation_Prompt, Reflect_Prompt
from .utils import GPT_QA, item_feature_to_str


seed = 42
random.seed(seed)
np.random.seed(seed)


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
    Update a persona through pairwise reflection.

    :param user: The information of the user.
    :param sequence: Pairs containing pos_item and neg_item dictionaries.
    :param learning_type: Use 'pairwise'; the legacy pointwise parser is retained.
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
