"""Optional clustering figures for experiment analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


def plot_clustering_heatmap(embeddings, class2index_list, distance_threshold=0.3, user_id='user', timestamp='timestamp'):
    distance_matrix = squareform(pdist(embeddings, metric='euclidean'))

    Z = linkage(embeddings, method='ward', metric='euclidean')

    dendro = dendrogram(Z, no_plot=True)
    ordered_indices = dendro['leaves']

    original_indices = np.arange(len(embeddings))

    ordered_original_indices = original_indices[ordered_indices]

    ordered_distance_matrix = distance_matrix[np.ix_(ordered_indices, ordered_indices)]


    color_map = plt.get_cmap('Set3', len(class2index_list))
    color_dict = {key: color_map(key%20) for key in class2index_list.keys()}

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        ordered_distance_matrix,
        annot=False,
        cmap='viridis',
        xticklabels=ordered_original_indices,
        yticklabels=ordered_original_indices,
        cbar_kws={'label': 'Distance'},
        linewidths=0.5,
        linecolor='black',
    )

    ax = plt.gca()


    index2class = {}
    for key, indices in class2index_list.items():
        for index in indices:
            index2class[index] = key


    for label in ax.get_xticklabels():
        index = int(label.get_text())
        class_value=index2class[index]
        label.set_color(color_dict[class_value])

    for label in ax.get_yticklabels():
        index = int(label.get_text())
        class_value=index2class[index]
        label.set_color(color_dict[class_value])

    plt.title('Euclidean Distance Heatmap (Ordered by Clusters)')

    os.makedirs(f'figure/{timestamp}', exist_ok=True)
    plt.savefig(f'figure/{timestamp}/heatmap_{user_id}.png')
    plt.clf()

    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=ordered_original_indices.astype(str), leaf_rotation=90, leaf_font_size=10)
    plt.title('Hierarchical Clustering Dendrogram')

    plt.savefig(f'figure/{timestamp}/dendrogram_{user_id}.png')
    plt.clf()

    print(f'Figures saved to figure/{timestamp}/')
