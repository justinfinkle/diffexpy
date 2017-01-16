__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

"""
This is the main script that runs different parts of the analysis
"""

import sys
import os
import time
import numpy as np
import pandas as pd

from cluster_analysis import ClusterAnalysis

def check_for_file_replacement(path):
    if os.path.isfile(path):
        print "File to save: ", path
        answer = raw_input('The proposed file already exists. Would you like to replace it [y/n]? ')
        if answer == 'y':
            replace = True

        elif answer == 'n':
            replace = False
        else:
            sys.exit('Invalid answer')
    else:
        replace = True
    return replace

# Variables
de_data_pickle = '../data/pickles/20150909_de_analysis_full_BGcorrected.pickle'    # Path to the processed data pickle
c_string = 'intersection_WT'        # Which data to use for clustering
fold_change = True                  # Use fold change
clustering_type = 'kmeans_dec'          # Clustering to use. ap, kmeans, or kmeans_dec
decimation_criteria = 0.5           # If using kmeans_dec, specify the average correlation criteria for stopping.
k_means_restarts = 10000            # Number of restarts for Kmeans
save_directory = '../clustering/'
ap_damping = 0.5
ap_preferences = np.arange(-500, 0, 100)
ap_preferences = np.concatenate((ap_preferences, np.arange(-90, 0, 10), np.arange(-9,0,1), np.arange(-0.9, 0, .1),
                                  np.arange(-.09,.01, .01)))
num_k = None                    # An array or list of numbers. Specifies the values of k to use for kmeans optimization
                                # Use none and supply an affinity pickle if using ap results to guide kmeans
previous_ap_pickle = save_directory+"20150910_ap.affinitypickle"


# Paths to save

# Confirm that the savepath exists
if not os.path.exists(save_directory):
    exit_string = 'The specified path (%s) to save files does not exists' %save_directory
    sys.exit(exit_string)

date = time.strftime('%Y%m%d')
if clustering_type == 'kmeans_dec':
    type_string = "_".join([date, clustering_type, str(decimation_criteria), str(k_means_restarts)])
    save_df_path = save_directory+type_string+'.kmeans_dec_pickle'

elif clustering_type == 'kmeans':
    type_string = "_".join([date, clustering_type, str(k_means_restarts)+'restarts'])
    save_df_path = save_directory+type_string+'.kmeans_pickle'

elif clustering_type == 'ap':
    type_string = "_".join([date, clustering_type])
    save_df_path = save_directory+"_".join([date, clustering_type, str(ap_damping)+'damping'])+'.affinity_pickle'
    ap_sim_path = save_directory+type_string

cluster_data_path = save_directory+type_string+'.datapickle'
replace = check_for_file_replacement(cluster_data_path)


# Confirm that all parameters are desired.
print('\nYou have specified to use "%s" for clustering.\nThe results will be saved to %s' %(clustering_type, save_df_path))
params_ok = raw_input("Would you like to continue [y/n]? ")
if params_ok != 'y':
    sys.exit("Clustering will not continue. Try again")

if replace:
    print('Calculating and writing data')
    # Read the pickled DEAnalysis object
    de = pd.read_pickle(de_data_pickle)

    if fold_change:
        stacked_data = de.stack_data_replicates(de.filtered_data)
        log_data = de.log_fold_change_data(stacked_data, remove_noise_value=False)
        average_data = de.make_average_data(de.stacked_to_data_dict(log_data))
    else:
        average_data = de.average_data.copy()

    cluster_data = average_data[c_string]

    # Save data used for clustering
    cluster_data.to_pickle(cluster_data_path)
else:
    print('Loading existing data for analysis')
    cluster_data = pd.read_pickle(cluster_data_path)

print "Starting cluster analysis..."
# Initialize cluster analysis object
ca = ClusterAnalysis(cluster_data)

if clustering_type == 'ap':
    # Perform Affinity Propagation Clustering optimization
    ap_df, sim_matrix = ca.preference_optimization_scan(preferences=ap_preferences, damping=ap_damping)
    print("saving ap results to %s" %save_df_path)
    ap_df.to_pickle(save_df_path)
    print("saving similarities used for ap to %s" %ap_sim_path)
    np.save(ap_sim_path, sim_matrix)

elif clustering_type == 'kmeans':
    # Perform KMeans clustering optimization
    if num_k is None:
        try:
            ap_df = pd.read_pickle(previous_ap_pickle)
            num_k = (ap_df['n_clusters'].values)
            num_k = list(set(num_k[num_k >= 1]))
            num_k.sort()
        except IOError as e:
            sys.exit(e)

    num_k = np.array(num_k)
    inertia, labels, centroids = ca.kmeans_optimization(num_k, k_means_restarts, show_output=True)
    k_means_df = pd.DataFrame([num_k, inertia, labels, centroids]).T
    k_means_df.columns = ['num_clusters','inertia', 'labels', 'centroids']
    print('saving kmeans results to %s' %save_df_path)
    k_means_df.to_pickle(save_df_path)

elif clustering_type == 'kmeans_dec':
    # Perform Kmeans clustering search by decimation
    correlation_k, decimation_df = ca.decimate_clusters(cluster_data)
    print('saving kmeans decimation results to %s' %save_df_path)
    decimation_df.to_pickle(save_df_path)

else:
    sys.exit("No valid clustering type selected. Choose a valid clustering type and rerun")