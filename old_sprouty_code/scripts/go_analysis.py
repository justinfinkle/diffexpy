__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import os
import cluster_analysis as ca
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.spatial import distance

cluster_data_file = '../clustering/20150910_kmeans_dec_0.5_10000.datapickle'
kmeans_decimation_results_file = '../clustering/20150910_kmeans_dec_0.5_10000.kmeans_dec_pickle'
scorer = ca.ClusterScorer(cluster_data_file, kmeans_results_path=kmeans_decimation_results_file)
opt_clustering = scorer.kmeans_results.iloc[-6]
labels = opt_clustering['labels'][0]
centroids = opt_clustering['centroids'][0]

# Save gene names for go analysis
gene_names = scorer.cluster_data.index.values
gene_names = [name.capitalize().replace('rik', 'Rik') for name in gene_names]
np.savetxt('../data/goa_data/filtered_genes.txt', gene_names, '%s')
scorer.cluster_data.index = gene_names


# Load KO data

# if it hasn't been pickled yet, use this
"""
de = pd.read_pickle('../data/pickles/20150909_de_analysis_full_BGcorrected.pickle')
c_string = 'intersection_KO'
stacked_data = de.stack_data_replicates(de.filtered_data)
log_data = de.log_fold_change_data(stacked_data)
average_data = de.make_average_data(de.stacked_to_data_dict(log_data))
cluster_data = average_data[c_string]
cluster_data.to_pickle('../clustering/20150915_ko_clustering_data.pickle')
sys.exit()
"""

# if it has been, just load the right pickle
ko_data = pd.read_pickle('../clustering/20150915_ko_clustering_data.pickle')

# Assign the KO data to the appropriate cluster
ko_labels = ca.assign_genes_to_cluster(ko_data, centroids)

# Make a dictionary of clusters
cluster_dict = {}
times = [0, 15, 60, 120, 240]
n_genes_in_same_cluster = 0
path = '../clustering/go_enrichment/ko_assigned_to_wt_clusters/'
for label in set(labels):
    cluster_dict[label] = {}
    current_wt_data = scorer.cluster_data[labels == label]
    plt.plot(times, current_wt_data.T)
    plt.show()
    current_ko_data = ko_data[ko_labels == label]
    current_centroid = centroids[label]
    current_wt_genes = current_wt_data.index.values
    current_ko_genes = current_ko_data.index.values
    cluster_dict[label]['WT'] = current_wt_data
    cluster_dict[label]['KO'] = current_ko_data
    cluster_dict[label]['WT_genes'] = current_wt_genes
    cluster_dict[label]['KO_genes'] = current_ko_genes
    overlapping_genes = set(current_wt_genes).intersection(current_ko_genes)
    ko_genes_not_in_wt = set(current_ko_genes).difference(current_wt_genes)
    cluster_dict[label]['gene_union'] = set(current_wt_genes).union(current_ko_genes)
    n_genes_in_same_cluster += len(overlapping_genes)
    current_gene_names = [name.capitalize().replace('rik', 'Rik') for name in list(cluster_dict[label]['gene_union'])]
    #np.savetxt(path+str(label)+'.txt', np.array(current_gene_names), fmt='%s')

# Test genes in each cluster for go enrichment
# Save each cluster of genes as a separate file


