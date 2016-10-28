__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import time
import re
import os
import pandas as pd
import numpy as np
import cluster_analysis as ca

# logFC corresponds to the log2-fold-change (see TopTable documentation for R)
p_val_threshold = 0.001
fc_thresh = 0.05
log_fc_threshold = np.log2(1+fc_thresh)
date = time.strftime('%Y%m%d')

# Filenames for saving results
save_results = True
base_folder = '../clustering/binary_cluster_attempts/'
binary_results_path = base_folder+'%s_binary_cluster_assignment_p%2.3f_fc%1.2f.pickle' % (date, p_val_threshold, fc_thresh)
cluster_dict_path = base_folder+'%s_binary_cluster_dictionary_p%2.3f_fc%1.2f.pickle' % (date, p_val_threshold, fc_thresh)
wt_path = base_folder+'%s_wt_binary_path_p%2.3f_fc%1.2f.pickle' % (date, p_val_threshold, fc_thresh)
ko_path = base_folder+'%s_ko_binary_path_p%2.3f_fc%1.2f.pickle' % (date, p_val_threshold, fc_thresh)

# Import data
r_results_folder = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/clustering/expression_change/"
files = os.listdir(r_results_folder)

# Read files into dataframe if it is a csv
expression_change_results = {filename.replace('.csv', ''): pd.read_csv(r_results_folder+filename) for filename in files
                             if ".csv" in filename}
comparisons = expression_change_results.keys()

# Make a list of the conditions
conditions = list(set([re.split(r"[_-]+", key)[0] for key in comparisons]))

#   Initialize the results
results_index = expression_change_results.itervalues().next().sort_index().index
results_dict = {condition: pd.DataFrame([], index=results_index) for condition in conditions}

# Assign each gene to a cluster
for key, df in expression_change_results.iteritems():

    # Get information to place in the appropriate dataframe
    key_split = re.split(r"[_-]+", key)
    condition = key_split[0]
    end_time = int(key_split[1])
    results_df = results_dict[condition]

    # Confirm that the list of genes matches
    df.sort_index(inplace=True)
    if not np.array_equal(results_df.index.values, df.index.values):
        raise Exception("List of sorted genes does not match reference ")

    results_df[end_time] = np.sign(df['logFC'].values)
    results_df.loc[df['adj.P.Val'] > p_val_threshold, end_time] = 0
    results_df.loc[np.abs(df['logFC']) < log_fc_threshold, end_time] = 0

# Clean up any gene mismatches due to R naming conventions
gene_list = results_index
gene_list = [name.capitalize().replace('rik', 'Rik')for name in gene_list]
gene_list = [name[1:] if (name[0] == 'X') & ('Rik' in name) else name for name in gene_list]
manual_edit = {'X732482': '732482', 'H2.dma': 'H2-dma', 'S3.12': 'S3-12'}
for wrong, right in manual_edit.iteritems():
    gene_list[gene_list.index(wrong)]=right

# Reformat index
for value in results_dict.itervalues():
    value.index = gene_list
    value.sort_index(inplace=True)
    value[0] = 0
    value.sort_index(axis=1, inplace=True, ascending=True)

# Now save the corresponding data for each cluster into a dictionary
########################################################################################################################
########################################################################################################################

combos = ca.make_binary_combos(['0', '1', '-1'], 4)

# Import data
de = pd.read_pickle('../data/pickles/protocol_2_de_analysis_full_BGcorrected.pickle')

stacked_data = de.stack_data_replicates(de.filtered_data)
log_data = de.stacked_to_data_dict(de.log_fold_change_data(stacked_data, remove_noise_value=False), replace_rep=True)

avg_data = de.make_average_data(log_data)
avg_dict = {'WT': avg_data['intersection_WT'], 'KO': avg_data['intersection_KO']}
for value in avg_dict.itervalues():
    value.index = [name.capitalize().replace('rik', 'Rik') for name in value.index.astype(str)]
    value.sort_index(inplace=True)

# Remove the cluster if it's just noisy for both WT and KO
# Clean up indices so that they match
keep_gene = (((results_dict['WT'].T != 0).any()) | ((results_dict['KO'].T != 0).any())).values
for condition in conditions:
    avg_dict[condition] = avg_dict[condition][keep_gene]
    results_dict[condition] = results_dict[condition][keep_gene]


if not np.array_equal(results_dict['WT'].index, avg_dict['WT'].index):
    raise ValueError('WT indices unequal')

if not np.array_equal(results_dict['KO'].index, avg_dict['KO'].index):
   raise ValueError('KO indicies unequal')

wt_trajectories = np.array([''.join(row.astype(int).astype(str)) for row in results_dict['WT'].values[:, 1:]])
ko_trajectories = np.array([''.join(row.astype(int).astype(str)) for row in results_dict['KO'].values[:, 1:]])

binary_df = pd.DataFrame([wt_trajectories, ko_trajectories], index=['WT', 'KO'], columns=results_dict['WT'].index.values).T
# print binary_df.loc[['Egr1', 'Fos', 'Spry2', 'Angptl4']]

cluster_dictionary = {}
times = [0, 15, 60, 120, 240]
show_plots = False
sse = 0
for combo in combos:
    wt_inc = avg_dict['WT'].iloc[wt_trajectories == combo]
    ko_inc = avg_dict['KO'].iloc[ko_trajectories == combo]
    current_wt_genes = wt_inc.index.values
    current_ko_genes = ko_inc.index.values
    intersection_genes = set(current_wt_genes).intersection(current_ko_genes)
    union_genes = set(current_wt_genes).union(current_ko_genes)
    plot_data = np.vstack((wt_inc.values, ko_inc.values)).T
    cluster_dictionary[combo] = {'wt_df': wt_inc, 'ko_df': ko_inc, 'wt_genes': current_wt_genes,
                                 'ko_genes': current_ko_genes, 'union_genes': union_genes,
                                 'intersection_genes': intersection_genes}

if save_results:
    pd.to_pickle(cluster_dictionary, cluster_dict_path)
    pd.to_pickle(binary_df, binary_results_path)
    pd.to_pickle(results_dict['WT'], wt_path)
    pd.to_pickle(results_dict['KO'], ko_path)

