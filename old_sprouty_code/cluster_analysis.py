# -*- coding: utf-8 -*-
__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import time
import itertools
import os
import sys
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy import stats
from scipy.stats import fisher_exact
from scipy.spatial import distance


def sum_of_squares(X):
        """
        Calculate the sum of the squares for each column
        :param X: array-like
            The data matrix for which the sum of squares is taken
        :return: float or array-like
            The sum of squares, columnwise or total
        """
        column_mean = np.mean(X, axis=0)
        ss = np.sum(np.power(X-column_mean, 2), axis=0)
        return ss


def cluster_correlation(data, labels, centroids):
    #todo: This does NOT like all zeros values
    """
    Calculate how well the genes in the cluster correlate with the centroid. For kmeans, the centroid
    is the same as the average cluster trajectory, by definition.
    :param data:
    :param labels:
    :return:
    """
    label_set = list(set(labels))
    label_set.sort()
    averages = cluster_average(data, labels)
    correlation_scores = np.zeros(len(label_set))

    for label in label_set:
        current_centroid = centroids[label]
        data_in_cluster = data[labels == label]
        try:
            corr = np.array([stats.pearsonr(current_centroid, gene)[0] for gene in data_in_cluster])
        except:
            data_in_cluster = data[labels == label].values
            corr = np.array([stats.pearsonr(current_centroid, gene)[0] for gene in data_in_cluster])
        correlation_scores[label] = np.mean(corr)
    return correlation_scores


def cluster_average(data, labels):
    label_set = set(labels)
    averages = np.array([np.mean(data[labels==label], axis=0) for label in label_set])
    return averages


def calculate_total_sse(data, labels, centroids):
    """
    Calculate the total sse for the clustering. This is the same as the inertia calculated in sklearn kmeans
    :param data:
    :param labels:
    :param centroids:
    :return:
    """
    label_set = set(labels)
    sse = 0
    for label in label_set:
        current_centroid = centroids[label]
        data_in_cluster = data[labels == label].values
        sse += np.sum(np.power(data_in_cluster-current_centroid, 2))
    return sse


def elbow_criteria(x,y):
    x = np.array(x)
    y = np.array(y)
    # Slope between elbow endpoints
    m1 = point_slope(x[0], y[0], x[-1], y[-1])
    # Intercept
    b1 = y[0] - m1*x[0]

    # Slope for perpendicular lines
    m2 = -1/m1

    # Calculate intercepts for perpendicular lines that go through data point
    b_array = y-m2*x
    x_perp = (b_array-b1)/(m1-m2)
    y_perp = m1*x_perp+b1

    # Calculate where the maximum distance to a line connecting endpoints is
    distances = np.sqrt((x_perp-x)**2+(y_perp-y)**2)
    index_max = np.where(distances==np.max(distances))[0][0]
    elbow_x = x[index_max]
    elbow_y = y[index_max]
    return elbow_x, elbow_y


def point_slope(x1,y1, x2,y2):
    slope = (y2-y1)/float(x2-x1)
    return slope


def assign_genes_to_cluster(new_data, centroids):
    """
    Return the appropriate centroid label for new data
    :param new_data:
    :param centroids:
    :return:
    """
    dist_to_centroids = distance.cdist(new_data, centroids)
    centroid_label = dist_to_centroids.argmin(axis=1)
    return centroid_label


def make_binary_combos(char_list, length, kind='char'):
    # Make a iterator
    combo_iterator = itertools.product(char_list, repeat=length)

    # Turn it into a list
    if kind == 'char':
        combos = [''.join(combo) for combo in combo_iterator]
    else:
        combos = [list(combo) for combo in combo_iterator]
    return combos


def make_binary_trajectories(log2_fc_data, axis=0, threshold=0.1):
    # Take the difference of the data, and remove NaN
    data_diff = log2_fc_data.diff(axis=axis)
    data_diff.fillna(0, inplace=True)

    # Set the threshold
    change_thresh = np.log2(1+threshold)

    # Binarize based on threshold
    data_diff[data_diff >= change_thresh] = 1
    data_diff[data_diff <= -change_thresh] = -1
    data_diff[np.abs(data_diff) < change_thresh] = 0

    # Make a list of the combination tuples
    diff_list = np.array([''.join(row.astype(int).astype(str)) for row in data_diff.values[:, 1:]])
    return diff_list, data_diff


class ClusterAnalysis(object):
    """
    An object that performs multiple types of clustering and identifies the best clustering solution
    """
    def __init__(self, data=None, similarity='-sqeuclidean'):
        """

        :param data: ndarray, (n_samples, n_features)
            The pre-prepared data to run clustering on. The number of variables (e.g. genes) to cluster as the rows and
             the number of features (i.e. measurements) for the variable as the columns.
        :return:
        """
        self.data = data
        self.similarity_metric = similarity
        if data is not None:
            self.sim_matrix = self.calc_sim_matrix(self.data, metric=self.similarity_metric)

    def preference_optimization_scan(self, sim_matrix=None, data=None, preferences=None, damping=0.95):
        if sim_matrix is None:
            sim_matrix = self.sim_matrix
        if data is None:
            data = self.data
        if preferences is None:
            max_sim = np.max(sim_matrix)
            min_sim = np.min(sim_matrix)
            med_sim = np.median(sim_matrix)
            preferences = np.linspace(min_sim, max_sim)

        print("Starting affinity propagation scan")
        print("Dataset includes %i datapoints" %len(sim_matrix))
        clusters = []
        silos = []
        labels = []
        cluster_center_indices = []
        for ii, pref in enumerate(preferences):
            print("\nPreference: %i of %i" %(ii+1, len(preferences)))
            tic = time.time()
            try:
                current_labels, center_idx = self.run_ap(sim_matrix, pref, damping)
                silo = metrics.silhouette_score(data, current_labels)
                print("Silhouette Coefficient: %0.3f" % silo)
                n_clusters_ = len(center_idx)
                clusters.append(n_clusters_)
                silos.append(silo)
                labels.append(current_labels.copy())
                cluster_center_indices.append(center_idx)
                print('Estimated number of clusters: %d' % n_clusters_)
            except:
                print("Error using this preference")
                print(sys.exc_info()[0])
                clusters.append(0)
                silos.append(-1)
                labels.append('NA')
                cluster_center_indices.append('NA')
            print('Time for AP: %0.2f min' % ((time.time()-tic)/float(60)))

        save_df = pd.DataFrame([preferences, clusters, silos, labels, cluster_center_indices]).T
        save_df.columns = ['Preference', 'num_clusters', 'Silhouette_Score', 'Labels', 'Center_Index']
        return save_df, self.sim_matrix

    def calc_sim_matrix(self, data, metric='-sqeuclidean'):
        """

        :param data: arraylike
            Rows are the points, and columns are the features (e.g. measurements)
        :param metric: string
        :return:
        """
        if metric == '-sqeuclidean':
            distances = -metrics.pairwise_distances(data, metric='sqeuclidean')
        else:
            distances = metrics.pairwise_distances(data, metric=metric)
        return distances

    def run_ap(self, sim_matrix, preference=None, damping=None):
        """
        Run affinity propagation. This is a wrapper around the sklearn method and REQUIRES a precomputed similarity
        matrix
        :param sim_matrix: array-like (n_samples, n_samples)
            The similarity matrix of the datapoints. Any pairwise metric can be used to calculate it. There is no
            default for this function, but the class uses -squared euclidean which is the default for AP
        :param preference: float, optional
            Default is the median of the similarity matrix. A lower preference will produce fewer exemplars (clusters)
        :param damping: float, optional
            Value between 0.5 and 1. Default is 0.5. From the original paper "Each message is set to lambda times its
            value from the previous iteration plus 1 – lambda times its prescribed updated value." Higher damping should
            produce a solution more robust to noise, but appears to take longer
        :return: tuple (labels, cluster center indices)
            The labels are the cluster each datapoint is assigned to, with length (n_samples). Cluster center indices
            are the indices for the exemplars chosen as the centers. The length depends on the preference chosen
        """
        if preference is None:
            preference = np.median(sim_matrix)
        if damping is None:
            damping = 0.5

        print("Fitting Affinity Propagation Model with: preference=%0.3f, damping=%0.2f"%(preference, damping))
        tic = time.time()
        ap = AffinityPropagation(preference=preference, damping=damping, affinity='precomputed')
        ap.fit(sim_matrix)
        print('Time for AP: %0.2f min' % ((time.time()-tic)/float(60)))
        return ap.labels_, ap.cluster_centers_indices_

    def kmeans_optimization(self, cluster_range, n_restarts, data=None, show_output=True):
        if data is None:
            data = self.data
        inertia = []
        labels = []
        centroids = []
        for ii, n_clusters in enumerate(cluster_range):
            if show_output:
                print("KMeans Run: %i of %i" %(ii, len(cluster_range)))
                print("k = %i" %n_clusters)
            tic = time.time()
            k = KMeans(n_clusters=n_clusters, n_init=n_restarts).fit(data)
            inertia.append(k.inertia_)
            labels.append(k.labels_)
            centroids.append(k.cluster_centers_)
            if show_output:
                print(time.time()-tic)

        #opt_cluster, opt_inertia = elbow_criteria(cluster_range, inertia)

        return np.array(inertia), np.array(labels), np.array(centroids)

    def decimate_clusters(self, data=None, threshold=0.75, min_step_size=1, n_restarts=10000):
        # Get maximum edges, assuming all explanors are also response variables and no self edges
        if data is None:
            data = self.data.copy()

        data = data.values

        [n, p] = data.shape

        max_clusters = n
        max_step_size = 10**int(np.log10(max_clusters))

        # Set ranges of step sizes, assumed to be powers of 10
        powers = int(np.log10(max_step_size/min_step_size))
        step_sizes = [max_step_size/(10**ii) for ii in range(powers+1)]

        # Intialize loop values
        cur_min = 2
        cur_max = max_clusters

        n_corr_list = []
        k_list = []
        label_list =[]
        centroid_list = []

        # Start stepping with forward like selection
        for ii, cur_step in enumerate(step_sizes):
            print("Current Step: ", cur_step)

            #Set the range to look through

            cur_range = np.arange(cur_min, cur_max, cur_step)
            if ii == 0 and max_clusters not in cur_range:
                cur_range = np.append(cur_range, max_clusters)

            # In the current range, check to see if every trace in the cluster correlates with the centroid
            # at the set threshold
            for cur_k in cur_range:
                if cur_k in k_list:
                    continue
                _, cur_labels, cur_centroids = self.kmeans_optimization([int(cur_k)], n_restarts,
                                                                        data=data, show_output=False)

                cur_scores = cluster_correlation(data, cur_labels[0], cur_centroids[0])
                num_corr = int(np.sum(cur_scores >= threshold))

                k_list.append(cur_k)
                label_list.append(cur_labels)
                n_corr_list.append(num_corr)
                centroid_list.append(cur_centroids)

                print("K: %i, Num_Corr: %i" % (cur_k, num_corr))
                if num_corr < cur_k:
                    cur_min = cur_k
                elif num_corr == cur_k:
                    # Found a new maximum that eliminates all betas, no need to keep stepping
                    cur_max = cur_k
                    break

        decimation_df = pd.DataFrame([k_list, n_corr_list, label_list, centroid_list]).T
        decimation_df.columns = ['num_clusters', 'n_correlated_clusters', 'labels', 'centroids']
        return cur_max, decimation_df

class ClusterScorer(object):
    """
    An object that reads clustering pickles and scores the clustering that was done
    """
    def __init__(self, cluster_data_path, ap_results_path=None, kmeans_results_path=None):
        self.cluster_data = pd.read_pickle(cluster_data_path)

        if ap_results_path is not None:
            self.ap_results = self.read_results(ap_results_path)
        if kmeans_results_path is not None:
            self.kmeans_results = self.read_results(kmeans_results_path)

    def read_results(self, path):
        df = pd.read_pickle(path)

        # Remove rows with missing data
        if 'Labels' in df.columns:
            df = df[df['Labels'] != 'NA']

        # Add centroids label to ap results
        if 'centroids' not in df.columns:
            centroids = [self.cluster_data.values[indices] for indices in df['Center_Index']]
            df['centroids'] = centroids

        return df

    def score_results(self, results_df, correlating_threshold=0.75):

        if 'Total_SSE' not in results_df.columns:
            print("Calculating Total SSE for each clustering. This may take several minutes\n")
            columns = [col.lower() for col in results_df.columns]
            labels_iloc = columns.index('labels')
            centroids_iloc = columns.index('centroids')
            sse_scores = [calculate_total_sse(self.cluster_data, results_df.iloc[row, labels_iloc],
                                              results_df.iloc[row, centroids_iloc]) for row in range(len(results_df))]

            results_df['Total_SSE'] = sse_scores

        if 'Silhouette_Score' not in results_df.columns:
            print("Calculating Silhouette Score for each clustering. This may take several minutes\n")
            columns = [col.lower() for col in results_df.columns]
            labels_iloc = columns.index('labels')
            s_scores = [metrics.silhouette_score(self.cluster_data, labelling) for labelling in results_df['labels']]
            results_df['Silhouette_Score'] = s_scores

        if 'num_correlated' not in results_df.columns:
            print("Calculating number of correlating trajectories for each clustering. This may take several minutes\n")
            columns = [col.lower() for col in results_df.columns]
            labels_iloc = columns.index('labels')
            centroids_iloc = columns.index('centroids')
            c_scores = [cluster_correlation(self.cluster_data, results_df.iloc[row, labels_iloc],
                                            results_df.iloc[row, centroids_iloc]) for row in range(len(results_df))]
            results_df['num_correlated'] = c_scores

        return

    def to_pickle(self, path):
        # Note, this is taken direclty from pandas generic.py which defines the method in class NDFrame
        """
        Pickle (serialize) object to input file path

        Parameters
        ----------
        path : string
            File path
        """
        from pandas.io.pickle import to_pickle
        return to_pickle(self, path)