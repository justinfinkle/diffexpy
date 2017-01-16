__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import pandas as pd
import numpy as np
import cluster_analysis as ca
from scipy import stats
import matplotlib.pyplot as plt
import pickle
import itertools
from sklearn import metrics


def plot_cluster(x_axis, plot_data, title=''):
    ci_color = np.array([123, 193, 238])/255.0
    plot_mean, plot_min, plot_max, plot_error = mean_confidence_interval(plot_data)
    plt.figure()
    plt.errorbar(x_axis[1:], plot_mean[1:],  yerr=plot_error[1:], fmt='o', lw=5, capsize=10, capthick=3, c=ci_color)
    plt.plot(x_axis, plot_mean, 'o-', ms=10, c=ci_color, lw=5)
    plt.legend(loc='best', prop={'size':18})
    plt.title(title, fontsize=24, weight='bold')
    plt.xlabel('Time after FGF Stimulation (min)', fontsize=20)
    plt.ylabel(r'$\log_2$(Fold Change)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.show()


def mean_confidence_interval(data, confidence=0.95, axis=1):
    #see: http://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data

    # Get the number of samples
    n = data.shape[axis]

    # Calculate the mean and standard error
    m, se = np.mean(data, axis=axis), stats.sem(data, axis=axis)
    std = np.std(data, axis=1, ddof=1)
    # Calculate the confidence interval range
    h = std #* stats.t.ppf((1+confidence)/2.0, n-1)
    return m, m-h, m+h, h


# Save pickle path
save_results = False
save_path = '../clustering/20150929_binary_cluster_assignment_0.05.pickle'
save_path2 = '../clustering/20150929_binary_clusters_0.05.pickle'
cluster_path = '../clustering/go_enrichment/20151002_binary_clusters_0.05/'

# Import data
de = pd.read_pickle('../data/pickles/protocol_2_de_analysis_full_BGcorrected.pickle')

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

diff = pd.DataFrame(ko_avg.values-wt_avg.values, columns=[0, 15, 60, 120, 250], index=wt_avg.index)
diff.drop([0], inplace=True, axis=1)
diff.to_csv('../data/drem_data/log_avg_ko_wt_intersection.tsv', sep='\t')
sys.exit()

data = pd.concat(de.filtered_data['intersection_WT'].values()+de.filtered_data['intersection_KO'].values(), axis=1)
#data.to_csv('data/intersection_genes_counts.csv')
#data = pd.concat(wt_data.values()+ko_data.values(), axis=1)
#data = pd.concat([wt_avg, ko_avg], axis=1)
#data.columns = [' '.join([col.split('-')[0], col.split('-')[2][:-1]]) for col in data.columns]
#data.to_excel('data/intersection_genes_counts.xlsx', encoding='ascii')

combos = ca.make_binary_combos(['0', '1', '-1'], 4)
# combo_iterator = np.array([list(c) for c in itertools.product([0, 1, -1], repeat=4)])
# sums = np.hstack((np.zeros((len(combo_iterator),1)), np.cumsum(combo_iterator, axis=1)))
# print sums.shape
# plt.plot(sums.T, 'o-', c='k')
# plt.show()
# sys.exit()
t = 0.05

wt_trajectories, _ = ca.make_binary_trajectories(wt_avg, axis=1, threshold=t)
ko_trajectories, _ = ca.make_binary_trajectories(ko_avg, axis=1, threshold=t)

top_genes = ['Pdzrn3', 'Cxcl1', 'Sncg', 'Adamts2', 'Wnt7b', 'Acta2', 'Cx3cl1', '1190002h23Rik', 'Rab15', 'Klk8',
             'Spry2', 'Spry3', 'Spry4', 'Nudt9', 'Arc', 'Pop4', 'Fosb', 'Siah2', 'Trib3', 'Gcap14', 'Serpina3m']


# Make sure the gene names match before saving the trajectories
if np.array_equal(wt_avg.index.values, ko_avg.index.values):
    binary_df = pd.DataFrame([wt_trajectories, ko_trajectories], index=['WT', 'KO'], columns=wt_avg.index.values).T

    if save_results:
        pd.to_pickle(binary_df, save_path)
        binary_df.to_csv(save_path.replace('pickle', 'csv'))

cluster_dictionary = {}
times = [0, 15, 60, 120, 240]
show_plots = False
for combo in combos:
    wt_inc = wt_avg.iloc[wt_trajectories == combo]
    ko_inc = ko_avg.iloc[ko_trajectories == combo]
    current_wt_genes = wt_inc.index.values
    current_ko_genes = ko_inc.index.values
    intersection_genes = set(current_wt_genes).intersection(current_ko_genes)
    union_genes = set(current_wt_genes).union(current_ko_genes)
    plot_data = np.vstack((wt_inc.values, ko_inc.values)).T
    cluster_dictionary[combo] = {'wt_df': wt_inc, 'ko_df': ko_inc, 'wt_genes': current_wt_genes,
                                 'ko_genes':current_ko_genes, 'union_genes': union_genes,
                                 'intersection_genes': intersection_genes}
    if save_results:
        #np.savetxt(cluster_path+combo+'.txt', list(union_genes), fmt='%s')
        np.savetxt(cluster_path+'WT/'+combo+'_WT.txt', list(current_wt_genes), fmt='%s')
        np.savetxt(cluster_path+'KO/'+combo+'_KO.txt', list(current_ko_genes), fmt='%s')
    if len(union_genes) > 0 and show_plots:
        plot_cluster(times, plot_data, title=combo)

if save_results:
    pd.to_pickle(cluster_dictionary, save_path2)