__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['font.sans-serif'] = 'Arial'

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

def plot_cluster(x_axis, plot_data, gene='', wt_trace=None, ko_trace=None, shape='', save_path=None):
    ci_color = np.array([123, 193, 238])/255.0
    #Scale plot data
    # plot_data = (plot_data/np.max(np.abs(plot_data), axis=0))
    # wt_trace = wt_trace/np.max(np.abs(wt_trace))
    # ko_trace = ko_trace/np.max(np.abs(ko_trace))
    plot_mean, plot_min, plot_max, plot_error = mean_confidence_interval(plot_data)
    plt.figure()
    #plt.errorbar(x_axis[1:], plot_mean[1:],  yerr=plot_error[1:], fmt='o', lw=5, capsize=10, capthick=3, c=ci_color)
    #Scale plot data
    # print plot_data.shape
    # print np.max(plot_data, axis=0).shape

    plt.plot(x_axis, plot_data, c='0.5')
    plt.plot(x_axis, plot_mean, 'o-', ms=10, c='k', lw=5, label='Cluster Average')
    if wt_trace is not None:
        plt.plot(x_axis, wt_trace, lw=2, c='b', label=(gene+' +Spry124'))
    if ko_trace is not None:
        plt.plot(x_axis, ko_trace, lw=2, c='r', label=(gene+' -Spry124'))

    plt.legend(loc='best', prop={'size': 18})
    title = 'Cluster containing %s [%s]' %(gene, shape)
    plt.title(title, fontsize=24, weight='bold')
    plt.xlabel('Time after FGF Stimulation (min)', fontsize=20)
    plt.ylabel('Percent of Max Fold Change', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlim([min(x_axis), max(x_axis)])
    plt.xticks(x_axis)
    # plt.ylim([-1.1, 1.1])
    plt.tight_layout()
    if save_path is None:
        print "None"
        plt.show()
    else:
        if os.path.isfile(save_path):
            answer = raw_input("File exists, do you want to replace it[y/n]? ")
            if answer=='y':
                plt.savefig(save_path, format='pdf')
            else:
                print "File not saved"
        else:
            plt.savefig(save_path, format='pdf')

def get_cluster_data(cluster_dict, binary_df, gene):
    wt_cluster_data = cluster_dict[binary_df.loc[gene, 'WT']]['wt_df']
    wt_trace = cluster_dict[binary_df.loc[gene, 'WT']]['wt_df'].loc[gene]
    ko_cluster_data = cluster_dict[binary_df.loc[gene, 'WT']]['ko_df']
    ko_trace = cluster_dict[binary_df.loc[gene, 'KO']]['ko_df'].loc[gene]
    return wt_cluster_data, wt_trace, ko_cluster_data, ko_trace

def plot_gene(gene, binary_df, cluster_dict, x_axis, de_obj):
    gene = gene.capitalize()
    if "RIK" in gene.upper():
        gene = gene.lower().replace('r', 'R')
    ci_color = np.array([123, 193, 238])/255.0
    clusters = binary_df.loc[gene].values
    wt_data, wt_trace, ko_data, ko_trace = get_cluster_data(cluster_dict, binary_df, gene)
    #n_subplots = len(set(clusters))
    n_subplots = 1
    f, axarr = plt.subplots(1, n_subplots)
    axarr = [axarr]
    if clusters[0] == clusters[1] and False:
        cluster = clusters[0]
        plot_data = np.vstack((wt_data.values, ko_data.values)).T
        plot_mean, plot_min, plot_max, plot_error = mean_confidence_interval(plot_data)
        axarr[0].errorbar(x_axis[1:], plot_mean[1:],  yerr=plot_error[1:], fmt='o', lw=5, capsize=10, capthick=3, c='purple')
        axarr[0].plot(x_axis, plot_mean, 'o-', ms=10, c='purple', lw=5, label='Cluster Average')
        #axarr[0].legend(loc='best', prop={'size':18})
        axarr[0].set_xlabel('Time after FGF Stimulation (min)', fontsize=24, weight='bold')
        axarr[0].set_ylabel(r'$\log_2$(Fold Change)', fontsize=24, weight='bold')
        axarr[0].tick_params(axis='both', which='major', labelsize=18)
        axarr[0].set_xticks(x_axis)
        axarr[0].set_title('Cluster [%s] Average Profile' %cluster, fontsize=24, weight='bold')
        wt_label = 'WT'
        ko_label = 'KO'
        axarr[1].plot(x_axis, wt_trace, 'o--', lw=3, c='b', alpha=0.5, label=wt_label)
        axarr[1].plot(x_axis, ko_trace, 'o--', lw=3, c='r', alpha=0.5, label=ko_label)
        axarr[1].legend(loc='best', prop={'size':18})
        #title = 'Cluster with shape [%s]' %cluster
        #axarr.set_title(title, fontsize=24, weight='bold')
        axarr[1].set_xlabel('Time after FGF Stimulation (min)', fontsize=24, weight='bold')
        axarr[1].tick_params(axis='both', which='major', labelsize=18)
        axarr[1].set_xticks(x_axis)
        axarr[1].legend(loc='best', fontsize=20)
        axarr[1].set_title(('%s Average Expression'%gene), fontsize=24, weight='bold')
        plt.tight_layout()
        plt.show()
        # plt.savefig(gene+'.pdf', format='pdf')
    else:
        reps = 'ABC'
        wt_raw = np.array([de_obj.filtered_data['intersection_WT'][rep].loc[gene.upper()].values for rep in reps]).T
        ko_raw = np.array([de_obj.filtered_data['intersection_KO'][rep].loc[gene.upper()].values for rep in reps]).T
        wt_mean, wt_min, wt_max, wt_error = mean_confidence_interval(wt_raw.astype(float))
        ko_mean, ko_min, ko_max, ko_error = mean_confidence_interval(ko_raw.astype(float))
        plot_data = wt_data.values.T
        plot_data = (plot_data/np.max(np.abs(plot_data), axis=0))
        wt_label = gene+' +Spry124'
        ko_label = gene+' -Spry124'
        axarr[0].errorbar(x_axis, wt_mean,  yerr=wt_error, fmt='o', lw=5, capsize=10, capthick=3, c='k')
        axarr[0].plot(x_axis, wt_mean, 'o-', ms=10, c='k', lw=5, label=wt_label)
        # ax2 = axarr[0].twinx()
        ax2 = axarr[0]
        ax2.errorbar(x_axis, ko_mean,  yerr=ko_error, fmt='o', lw=5, capsize=10, capthick=3, c='m')
        ax2.plot(x_axis, ko_mean, 'o-', ms=10, c='m', lw=5, label=ko_label)
        ax2.legend(loc='best', prop={'size': 20})
        ax2.set_xlabel('Time after FGF Stimulation (min)', fontsize=20, weight='bold')
        ax2.set_ylabel("Quantile Normalized Counts", fontsize=20, weight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.set_xticks(x_axis)
        ax2.set_title(('%s Average Expression'%gene), fontsize=24, weight='bold')
        # plot_mean, plot_min, plot_max, plot_error = mean_confidence_interval(plot_data)
        # axarr[1].errorbar(x_axis[1:], plot_mean[1:],  yerr=plot_error[1:], fmt='o', lw=5, capsize=10, capthick=3, c='b')
        # label = 'WT [%s]' %binary_df.loc[gene]["WT"]
        # axarr[1].plot(x_axis, plot_mean, 'o-', ms=10, c='b', lw=5, label=label)
        # plot_data = ko_data.values.T
        # plot_data = (plot_data/np.max(np.abs(plot_data), axis=0))
        # plot_mean, plot_min, plot_max, plot_error = mean_confidence_interval(plot_data)
        # axarr[1].errorbar(x_axis[1:], plot_mean[1:],  yerr=plot_error[1:], fmt='o', lw=5, capsize=10, capthick=3, c='r')
        # label = 'KO [%s]' %binary_df.loc[gene]["KO"]
        # axarr[1].plot(x_axis, plot_mean, 'o-', ms=10, c='r', lw=5, label=label)
        # axarr[1].set_xlabel('Time after FGF Stimulation (min)', fontsize=20, weight='bold')
        # axarr[1].tick_params(axis='both', which='major', labelsize=18)
        # axarr[1].set_xticks(x_axis)
        # axarr[1].legend(loc='best', fontsize=20)
        # axarr[1].set_title('Cluster Average Profile', fontsize=24, weight='bold')
        # axarr[1].set_ylabel('Percent of Max Fold Change', fontsize=20, weight='bold')
        plt.tight_layout()
        plt.show()
        # plt.savefig(("%s_expression.pdf"%gene), format='pdf')

def plot_traces(ax_obj, x_axis, traces, labels, **kwargs):
    plot_lines = []
    for ii, (trace, label) in enumerate(zip(traces, labels)):
        if ii==1 and len(traces)==2:
            twin_ax = ax_obj.twinx()
            new_line, = twin_ax.plot(x_axis, trace, label=label, color='r', **kwargs)
            for tl in twin_ax.get_yticklabels():
                tl.set_color('r')
        else:
            new_line, = ax_obj.plot(x_axis, trace, label=label, **kwargs)

        plot_lines.append(new_line)

    legend_labels = [l.get_label() for l in plot_lines]
    ax_obj.legend(plot_lines, legend_labels, loc='best')

    return ax_obj

binary_path = '../clustering/binary_cluster_attempts/20151111_binary_cluster_assignment_0.05.pickle'
cluster_dict_path = '../clustering/binary_cluster_attempts/20151111_binary_cluster_dictionary_0.05.pickle'

binary_df = pd.read_pickle(binary_path)
#print binary_df[(binary_df.WT=='11-1-1')&(binary_df.KO=='11-1-1')].sort('KO')
cluster_dict = pd.read_pickle(cluster_dict_path)
de = pd.read_pickle('../data/pickles/protocol_2_de_analysis_full_BGcorrected.pickle')

gene_of_interest = 'egr1'
# print binary_df.loc[gene_of_interest.capitalize()]
# sys.exit()
times = [0, 15, 60, 120, 240]
plot_gene(gene_of_interest, binary_df, cluster_dict, times, de)
# wt_d, wt_t, ko_d, ko_t = get_cluster_data(cluster_dict, binary_df, gene_of_interest)
# p_data = np.vstack((wt_d.values, ko_d.values)).T
# shape = binary_df.loc[gene_of_interest].values[0]
# plot_cluster(times, p_data, gene_of_interest, wt_t, ko_t, shape)#, '../clustering/20151005_binary_cluster_fos.pdf')

sys.exit()
#genes = ['nudt9', 'pop4', 'gcap14', 'siah2', 'arc', 'fosb']
genes = ['spry4', 'egr2', 'fosb']
genes = [gene.capitalize() for gene in genes]
wt_traces = []
ko_traces = []
labels = []
f, ax = plt.subplots(1, 2, figsize=(15, 7.5))
for gene in genes:
    _, wt_trace, _, ko_trace = get_cluster_data(cluster_dict, binary_df, gene)
    wt_traces.append(wt_trace.values)
    ko_traces.append(ko_trace.values)

ax1 = plot_traces(ax[0], times, ko_traces, genes, lw=3, marker='o')
ax2 = plot_traces(ax[1], times, wt_traces, genes, lw=3, marker='o')

plt.tight_layout()
plt.show()


sys.exit()
condition = 'WT'
cluster_string = binary_df.loc[gene_of_interest, condition]
print cluster_string
plot_data = np.vstack((cluster_dict[cluster_string][condition.lower()+'_df'].values)).T
print(plot_data.shape)

title = " ".join(['Cluster containing', gene_of_interest, condition, "["+cluster_string+"]"])
plot_cluster(times, plot_data, title)

