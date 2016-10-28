__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import cluster_analysis as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
cluster_data_file = '../clustering/intersection_WT/log_fc/affinity_propagation/clustering_data.pickle'
ap_cluster_results_file = '../clustering/intersection_WT/log_fc/affinity_propagation/ap_results.pickle'
kmeans_cluster_results_file = '../clustering/intersection_WT/log_fc/k_means/k_means_results_10k.pickle'
scorer = ca.ClusterScorer(cluster_data_file, ap_cluster_results_file, kmeans_cluster_results_file)
scorer.score_results(scorer.ap_results)
scorer.score_results(scorer.kmeans_results)
scorer.to_pickle('../clustering/intersection_WT/log_fc/scored_results.pickle')
"""

cluster_data_file = '../clustering/intersection_WT/log_fc/affinity_propagation/clustering_data.pickle'
scorer = ca.ClusterScorer(cluster_data_file)
previous_scorer = pd.read_pickle('../clustering/intersection_WT/log_fc/scored_results.pickle')
scorer.ap_results = previous_scorer.ap_results.copy()
scorer.kmeans_results = previous_scorer.kmeans_results.copy()
#scorer.score_results(scorer.ap_results)
scorer.score_results(scorer.kmeans_results)

plt.plot(scorer.ap_results['n_clusters'], scorer.ap_results['Total_SSE'], '.',
         scorer.kmeans_results['num_clusters'], scorer.kmeans_results['Total_SSE'], '.')
plt.legend(['AP', 'Kmeans'])
plt.show()

plt.plot(scorer.ap_results['n_clusters'], scorer.ap_results['Silhouette_Score'], '.',
         scorer.kmeans_results['num_clusters'], scorer.kmeans_results['Silhouette_Score'], '.')
plt.legend(['AP', 'Kmeans'])
plt.show()