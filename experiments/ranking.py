"""Downstream candidate ranking helpers used by the experiments."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from personax.utils import item_feature_to_str
from .rerank import compute_score


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
