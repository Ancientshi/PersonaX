"""Ranking metrics used by the experiment client."""

import numpy as np


def ndcg(test_truth_list, test_prediction_list, topk):
    ndcgs = []

    for k in topk:
        ndcg_list = []

        for ind, test_truth in enumerate(test_truth_list):
            dcg = 0
            idcg = 0
            test_truth_index = set(test_truth)

            if len(test_truth_index) == 0:
                continue

            top_sorted_index = test_prediction_list[ind][0:k]

            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    dcg += 1.0 / np.log2(index + 2)  # index从0开始，因此 +2

            sorted_truth_index = list(test_truth)[:k]
            for ideal_index in range(min(len(sorted_truth_index), k)):
                idcg += 1.0 / np.log2(ideal_index + 2)

            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0

            ndcg_list.append(ndcg)

        if len(ndcg_list) > 0:
            ndcgs.append(np.mean(ndcg_list))
        else:
            ndcgs.append(0.0)

    return ndcgs

def hit(test_truth_list, test_prediction_list, topk):
    hits = []

    for k in topk:
        hit_list = []

        for ind, test_truth in enumerate(test_truth_list):
            test_truth_set = set(test_truth)

            if len(test_truth_set) == 0:
                continue

            top_sorted_index = test_prediction_list[ind][0:k]

            hit = 1.0 if any(item in test_truth_set for item in top_sorted_index) else 0.0
            hit_list.append(hit)

        if len(hit_list) > 0:
            hits.append(np.mean(hit_list))
        else:
            hits.append(0.0)

    return hits

def mrr(test_truth_list, test_prediction_list, topk):
    mrrs = []

    for k in topk:
        mrr_list = []

        for ind, test_truth in enumerate(test_truth_list):
            test_truth_set = set(test_truth)

            if len(test_truth_set) == 0:
                continue

            top_sorted_index = test_prediction_list[ind][0:k]

            mrr = 0.0
            for index, item in enumerate(top_sorted_index):
                if item in test_truth_set:
                    mrr = 1.0 / (index + 1)
                    break

            mrr_list.append(mrr)

        if len(mrr_list) > 0:
            mrrs.append(np.mean(mrr_list))
        else:
            mrrs.append(0.0)

    return mrrs
