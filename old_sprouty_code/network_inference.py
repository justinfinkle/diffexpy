#!/Users/jfinkle/anaconda/bin/python
__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import os
import time
import itertools
import numpy as np
import pandas as pd
import pickle
from dionesus import Dionesus
import discretized_clustering as dcluster
from sklearn.decomposition import PCA
import cluster_analysis as ca
import network_analysis as na
from scipy import stats
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['font.sans-serif'] = 'Arial'
import matplotlib.pyplot as plt


def permute_data(array):
    """Warning: Modifies data in place. also remember the """
    # Array samples are rows, must transpose to columns
    new_array = array.copy().T
    _ = [np.random.shuffle(i) for i in new_array]
    return new_array.T

def save_permutes(x, num_permutes, path):
    for ii in range(num_permutes):
        x_permuted = permute_data(x)
        np.save(path+"/permuted_%i" %ii, x_permuted)

def fast_low_mem_ptest(n_pcs, p, o_yMatrix, original_betas, original_importance, permutepath):
    zeros = np.zeros(original_betas.shape)
    beta_result = {'n':zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}
    importance_result = {'n':zeros.copy(), 'mean': zeros.copy(), 'ss': zeros.copy()}
    d = Dionesus(n_pcs)
    for perm in range(p):
        tic = time.time()
        print "Permutation %i of %i" %(perm, p)
        cur_path = permutepath+"/permuted_%i.npy" %perm
        cur_x = np.load(cur_path)

        d.run_dionesus(cur_x, o_yMatrix, show_output=False)
        perm_betas = [np.array(d.beta_matrix.copy())]
        perm_importance = [np.array(d.importance_matrix.copy())]
        beta_result = update_variance_2D(beta_result, perm_betas)
        importance_result = update_variance_2D(importance_result, perm_importance)
        print time.time()-tic

    beta_stdev = np.sqrt(beta_result['variance'])
    importance_stdev = np.sqrt(importance_result['variance'])

    # Calculate zscores
    beta_zscores = (original_betas-beta_result['mean'])/beta_stdev
    importance_zscores = (original_importance-importance_result['mean'])/importance_stdev

    # Convert to p-value
    beta_p = 2 * stats.norm.cdf((-1*abs(beta_zscores)))
    importance_p = 2 * stats.norm.cdf((-1*abs(importance_zscores)))

    return beta_p, importance_p

def update_variance_2D(prev_result, new_samples):
        """incremental calculation of means: accepts new_samples, which is a list of samples. then calculates a new mean. this is a useful function for calculating the means of large arrays"""
        n = prev_result["n"] #2D numpy array with all zeros or watev
        mean = prev_result["mean"] #2D numpy array
        sum_squares = prev_result["ss"] #2D numpy array

        #new_samples is a list of arrays
        #x is a 2D array
        for x in new_samples:
            n = n + 1
            #delta = float(x) - mean
            old_mean = mean.copy()
            mean = old_mean + np.divide( (x-old_mean) , n)
            sum_squares = sum_squares + np.multiply((x-mean),(x-old_mean))

        if (n[0,0] < 2):
            result = {  "mean": mean,
                        "ss": sum_squares,
                        "n": n}
            return result

        variance = np.divide(sum_squares,(n-1))
        result = {  "mean": mean,
                    "ss": sum_squares,
                    "variance": variance,
                    "n": n}
        return result

def save_pca_figure(data):
    #todo: This is a crummy function. Fix it
    date = time.strftime('%Y%m%d')
    # Do PCA to see explained variance spread
    pca = PCA()
    pca_data = data.T
    pca.fit(pca_data)
    max_pc, _ = ca.elbow_criteria(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    max_components = max_pc+1
    transformed_data = pca.transform(pca_data)

    scores_labels = pca_data.index.values
    pca_index = [index for index in range(len(scores_labels)) if '-0' not in scores_labels[index]]

    x_pc = 0
    y_pc = 1
    plt.figure(figsize=(10,10))
    plt.scatter(pca.components_[x_pc, :], pca.components_[y_pc, :])
    thresh = 0.1
    for ii, xy in enumerate(zip(pca.components_[x_pc, :], pca.components_[y_pc, :])):
        if (abs(xy[0])>thresh or abs(xy[1])>thresh):
            plt.annotate(data.index.values[ii], xy=xy, fontsize=16)
    plt.show()

    sys.exit()
    plt.figure(figsize=(15,15))
    plt.scatter(transformed_data[pca_index, x_pc], transformed_data[pca_index, y_pc],
                linewidth=8, edgecolors='b')
    manual_labels = ['KO-25-120B', 'KO-22-60A', 'WT-7-60B', 'WT-7-60C', 'WT-10-120A', 'KO-19-15A','KO-19-15C']
    for ii, xy in enumerate(zip(transformed_data[pca_index, x_pc], transformed_data[pca_index, y_pc])):
        if scores_labels[pca_index[ii]] in manual_labels:
            xy = (xy[0]-2, xy[1]-1.5)
        else:
            xy = (xy[0]+0.5, xy[1]+0.5)
        plt.annotate(scores_labels[pca_index[ii]], xy=xy, fontsize=16)
    plt.xlabel('Principal Component '+str(x_pc+1), fontsize=28, weight='bold')
    plt.ylabel('Principal Component '+str(y_pc+1), fontsize=28, weight='bold')
    plt.xlim([-22, 22])
    plt.ylim([-22, 22])
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.axhline(linewidth=4, color="k")
    plt.axvline(linewidth=4, color="k")
    save_string = '../network_inference/%s_sprouty_pca%i_%i.pdf'%(date,x_pc+1,y_pc+1)
    plt.savefig(save_string, format='pdf')

# Import data
de = pd.read_pickle('data/pickles/protocol_2_de_analysis_full_BGcorrected.pickle')

#Read old files
# pc8 = pd.read_csv('network_inference/20151001_dionesus_8pc_log_fc_intersection_all_link_list.csv')
# pc8.drop(['Unnamed: 0'], axis=1, inplace=True)
# pc8 = pc8[pc8['VIP_pval']<=0.05]
# pc8_edges = list(zip(pc8.Parent, pc8.Child))
# pc2 = pd.read_csv('network_inference/20150912_dionesus_2pc_log_fc_intersection_all_link_list.csv')
# pc2.drop(['Unnamed: 0'], axis=1, inplace=True)
# pc2 = pc2[pc2['VIP_pval']<=0.05]
# pc2['Edge'] = zip(pc2.Parent, pc2.Child)
# binary = [(edge in pc8_edges) for edge in pc2.Edge.values]
# merged_df = pc2.loc[binary].copy()
# merged_df.drop('Edge', axis=1, inplace=True)
# # na.save_gdf(merged_df, 'network_inference/20151006_dionesus_2pc_8pc_merged.gdf')
# sys.exit()

c_string = 'intersection_KO'
stacked_data = de.stack_data_replicates(de.filtered_data)
log_data = de.stacked_to_data_dict(de.log_fold_change_data(stacked_data, remove_noise_value=False), replace_rep=True)
wt_data = log_data['intersection_WT']
ko_data = log_data['intersection_KO']


avg_data = de.make_average_data(log_data)
wt_avg = avg_data['intersection_WT']
wt_avg.index = [name.capitalize().replace('rik', 'Rik') for name in wt_avg.index.values]
ko_avg = avg_data['intersection_KO']
ko_avg.index = [name.capitalize().replace('rik', 'Rik') for name in ko_avg.index.values]

#data = pd.concat(de.filtered_data['intersection_WT'].values()+de.filtered_data['intersection_KO'].values(), axis=1)
#data.to_csv('data/intersection_genes_counts.csv')
#data = pd.concat(wt_data.values()+ko_data.values(), axis=1)
#data = pd.concat([wt_avg, ko_avg], axis=1)
#data.columns = [' '.join([col.split('-')[0], col.split('-')[2][:-1]]) for col in data.columns]
#data.to_excel('data/intersection_genes_counts.xlsx', encoding='ascii')


# Combine all data into one data frame
data = pd.concat(wt_data.values()+ko_data.values(), axis=1)

# Filter out genes that don't have a significant difference between WT and KO at ANY time points
dea_path = 'data/pickles/protocol_2_de_analysis_full_BGcorrected.pickle'
dc_path = 'data/pickles/'
r_results_path_diff = 'clustering/differential_expression/'
fdr = 0.05
p_thresh = 0.05
fc_thresh = 0.1
dc_diff = dcluster.DiscretizedClusterer(dea_path, r_results_path_diff, p_value_threshold=p_thresh,
                                        fold_change_threshold=fc_thresh)
boolean = pd.DataFrame()
for comp, limma in dc_diff.limma_results.iteritems():
    boolean = pd.concat([boolean, (limma['adj.P.Val'] < p_thresh) & (np.abs(limma['logFC']) >= fc_thresh)], join='outer', axis=1)

# Add back in the sprouty genes. It doesn't make sense to remove them because the dynamics after knockout don't make sense
boolean.loc[['SPRY2', 'SPRY3', 'SPRY4']] = True
boolean = boolean[np.sum(boolean, axis=1) > 0]
boolean.index = [idx.replace('99384', 'X99384') for idx in boolean.index]
filtered_data = data.join(boolean, how='inner').iloc[:, :-boolean.shape[1]]
#save_pca_figure(data)

# Run dionesus
# Save permutes
data = filtered_data
num_permutes = 1000
xMatrix = data.T.values
yMatrix = data.T.values
permute_directory = 'network_inference/filtered_permutes/'
permutes_in_directory = len(os.listdir(permute_directory))

if permutes_in_directory < num_permutes:
    print('Saving permute data')
    save_permutes(xMatrix, num_permutes, permute_directory)

num_pcs = 2 # Use two PCs because this is how data splits for Time and WT/KO data. See sprouty_pca1_2.pdf
dio = Dionesus(num_pcs)
dio.run_dionesus(xMatrix, yMatrix, show_output=False)
dio.explanatory_variables = data.index.values
dio.response_variables = data.index.values

save_file_path = 'network_inference/20160617_prefiltered_dionesus_2pc_log_fc_intersection'
bp, ip = fast_low_mem_ptest(num_pcs,num_permutes, yMatrix, dio.beta_matrix, dio.importance_matrix, permute_directory)
pickle.dump(dio, open(save_file_path+'.pickle', 'wb'))
ll = dio.create_link_list(vip_pvals=ip, beta_pvals=bp, vip_threshold=1)
x, y = ca.elbow_criteria(range(len(ll.VIP.astype(float))), ll.VIP.astype(float))
# filtered_ll = ll[ll.VIP>1]
ll.to_csv(save_file_path+'_link_list.csv')

na.save_gdf(ll, save_file_path+'.gdf')



