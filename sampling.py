import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import pdist, squareform
import math
seed=42
random.seed(seed)
np.random.seed(seed)


def select_samples(points, alpha=1.1, num_samples=10):
    center= points.mean(axis=0)
    size=len(points)
    ratio=num_samples/size
    
    def cal_aim(c,point,selected_points):
        w_p = alpha ** (-10)
        w_d = 1-w_p

        distance=np.linalg.norm(point-center)
        easy=w_p*1/(1+distance)
        
        if len(selected_points)==0:
            if_selected_points=np.array([point])
        else:
            if_selected_points= np.concatenate((selected_points, [point]), axis=0)

        #Calculate the distance sum of two points in if_selected_points
        distance_matrix = squareform(pdist(if_selected_points, metric='euclidean'))
        distance_matrix = np.triu(distance_matrix, k=1)
        distance_avg= np.sum(distance_matrix)/len(if_selected_points)
        diversity=w_d * distance_avg
        score= easy+diversity
        return score, easy, diversity
    
    selected_points=[]
    selected_indexs=[]
    selected_scores=[]
    for i in range(num_samples):
        scores=[cal_aim(c,point,selected_points) for c,point in enumerate(points)]
        sum_score=[score[0] for score in scores]
        easy_score=[score[1] for score in scores]
        diversity_score=[score[2] for score in scores]
        
        valid_indices = [j for j in range(len(sum_score)) if j not in selected_indexs]
        max_valid_index = np.argmax(np.take(sum_score, valid_indices))
        max_index = valid_indices[max_valid_index]

        selected_points.append(points[max_index])
        selected_indexs.append(max_index)
        selected_scores.append(scores[max_index])

    return np.array(selected_points),selected_indexs,selected_scores




def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_allocation(cluster_size_list, budget):
    #The budget must be at least equal to or greater than the number of clusters, so that each cluster can be allocated at least one
    if budget<len(cluster_size_list):
        budget=len(cluster_size_list)
        
    # Convert to numpy array for easier handling
    cluster_size_list = np.array(cluster_size_list)
    
    # Sort cluster sizes in ascending order and get their sorted indices
    cluster_size_list_index = np.argsort(cluster_size_list)
    sorted_cluster_size_list = cluster_size_list[cluster_size_list_index]
    
    # Initialize allocation list
    allocation_list = []
    
    # Distribute the budget
    for i in range(len(sorted_cluster_size_list)):
        unallocated_strata = len(sorted_cluster_size_list) - len(allocation_list)
        avg = budget // unallocated_strata
        cluster_size = sorted_cluster_size_list[i]
        
        # Allocate budget
        if cluster_size <= avg:
            allocation_list.append(cluster_size)
        else:
            allocation_list.append(avg)
        
        # Update the remaining budget
        budget -= allocation_list[-1]
    
    if budget > 0:
        for i in range(budget):
            allocation_list[-(i+1)] += 1
    
    # Restore the allocation list to the original order
    final_allocation_list = [0] * len(cluster_size_list)
    for i, index in enumerate(cluster_size_list_index):
        final_allocation_list[index] = allocation_list[i]
    
    return final_allocation_list

    
    


def sampling(cluster_list,alpha=1.1, ratio=0.6):
    sum_size=sum([len(cluster) for cluster in cluster_list])
    budget=math.ceil(sum_size*ratio)
    cluster_size_list=[len(cluster_list[i]) for i in range(len(cluster_list))]
    allocation=get_allocation(cluster_size_list,budget)

    selected_indexs_list=[]
    for i in range(len(cluster_list)):
        cluster=cluster_list[i]
        selected_points, selected_indexs,selected_scores =select_samples(cluster,alpha,allocation[i])
        selected_indexs_list.append(selected_indexs)
        
    return selected_indexs_list



def relevance_sampling(pos_item_embeddings,target_item_embedding,ratio):
    relevance_scores=np.dot(pos_item_embeddings,target_item_embedding)
    num_samples=math.ceil(len(relevance_scores)*ratio)
    selected_indexs=np.argsort(relevance_scores)[-num_samples:]
    return selected_indexs
