__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import fisher_test as ft
import pandas as pd
import numpy as np
import sys
import time
import os
import matplotlib.pyplot as plt
import cluster_analysis as ca


def make_possible_paths(step_list, step_values):
    """

    :param step_list:
    :param step_values:
    :return:
    """
    stepwise_path_combos = {}
    string_to_array_dict = {}
    for step in steps:
        step_combos = ca.make_binary_combos([0, 1, -1], step, kind='arr')
        stepwise_path_combos[step] = {'0' + ''.join(map(str, combo)): np.array([0] + combo) for combo in step_combos}
        for combo in step_combos:
            string_to_array_dict['0' + ''.join(map(str, combo))]= np.array([0] + combo)
    return string_to_array_dict, stepwise_path_combos


def binary_to_growing_path(step_list, df):
    """

    :param df:
    :return:
    """
    growing_path = pd.DataFrame([], index=df.index)

    for step in step_list:
        a = df.values[:, :step+1].astype(int).astype(str)
        a = [''.join(row) for row in a.astype(str)]
        growing_path[step] = a

    return growing_path


def find_genes_in_path(growing_path_df, stepwise_path_combos):

    path_dictionary = {}
    for step, series in growing_path_df.iteritems():
        path_set = set(series)
        path_dictionary[step] = {}
        genes_in_step = 0
        possible_paths = stepwise_path_combos[step]
        for path in path_set:
            path_array = possible_paths[path]
            alt_paths = [alt_path for alt_path in path_set if
                         (alt_path != path) & np.array_equal(possible_paths[alt_path][:step], path_array[:step])]
            genes_in_path = growing_path.index[growing_path[step]==path].values
            tfs_in_path = ft.convert_gene_to_tf(genes_in_path, gene_dict)
            path_dictionary[step][path] = {'genes': genes_in_path.tolist(), 'tfs': tfs_in_path, 'alt_paths':alt_paths}
            genes_in_step += len(genes_in_path)
        if genes_in_step != len(growing_path):
            raise ValueError('Not all genes accounted for in step %i' % step)

    return path_dictionary


def find_path_enrichment(path_dictionary, fdr=0.01):

    enriched_path_list = []
    for step, paths in path_dictionary.iteritems():
        for path, value in paths.iteritems():
            study_tfs = value['tfs']
            alts = value['alt_paths']
            alt_tfs = []
            for alt in alts:
                alt_tfs += path_dictionary[step][alt]['tfs']
            table = ft.calculate_study_enrichment(study_tfs, alt_tfs, fdr=fdr)

            if np.sum(table['FDR_reject']) > 0:
                print path
                print table[table['FDR_reject']]
                print 
                enriched_path_list.append(path)
    return enriched_path_list


def plot_enrichment_results(enriched_path_list, string_to_array_dict):
    print len(enriched_path_list)
    n_cols = int(np.floor(np.sqrt(len(enriched_path_list))))
    n_rows = int(np.ceil(np.sqrt(len(enriched_path_list))))
    # f, axarr = plt.subplots(n_rows, n_cols)

    a, b = np.meshgrid(range(n_rows), range(n_cols))
    row_index = a.flatten()
    column_index = b.flatten()
    for ii, sig in enumerate(enriched_path_list):
        found = [string.find(sig) for string in enriched_path_list]
        if np.sum(found)>-len(enriched_path_list)+1:
            print sig
            continue
        plot_data = string_to_array_dict[sig]
        x_ticks = range(len(plot_data))
        max_mag = np.max(np.abs(np.cumsum(plot_data)))
        plt.plot(np.cumsum(plot_data), '.-', lw=3, ms=20, label=sig)
        #plt.plot(x_ticks[-2], plot_data[-2], '.', lw=3, ms=20, c='r')
        # plot_data = string_to_array_dict[sig]
        # max_mag = np.max(np.abs(np.cumsum(plot_data)))
        # x_ticks = range(len(plot_data))
        # axarr[row_index[ii], column_index[ii]].plot(x_ticks, plot_data, '.-', lw=3, ms=20)
        # axarr[row_index[ii], column_index[ii]].plot(x_ticks[-2:], plot_data[-2:], '.-', lw=3, ms=20, c='r')
        # axarr[row_index[ii], column_index[ii]].set_ylim([-max_mag, max_mag])
        # axarr[row_index[ii], column_index[ii]].set_yticks(np.arange(-max_mag, max_mag+1))
        # axarr[row_index[ii], column_index[ii]].set_xticks(x_ticks)
        # axarr[row_index[ii], column_index[ii]].set_xticklabels(times[:len(plot_data)])
    #plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    p_val_threshold = 0.001
    fc_thresh = 0.05
    log_fc_threshold = np.log2(1+fc_thresh)
    date = time.strftime('%Y%m%d')
    date = '20151112'

    base_folder = '../clustering/binary_cluster_attempts/'
    association_file = '../clustering/tf_enrichment/mouse_predicted.txt'
    association_dict_file = '../clustering/tf_enrichment/gene_association_dict.pickle'
    gene_set = '../data/goa_data/mus_musculus_gene_set'
    binary_path = base_folder+'%s_binary_cluster_assignment_p%2.3f_fc%1.2f.pickle' % (date, p_val_threshold, fc_thresh)
    wt_path = base_folder+'%s_wt_binary_path_p%2.3f_fc%1.2f.pickle' % (date, p_val_threshold, fc_thresh)
    ko_path = base_folder+'%s_ko_binary_path_p%2.3f_fc%1.2f.pickle' % (date, p_val_threshold, fc_thresh)
    tf_per_node_file = '../clustering/tf_enrichment/tf_per_node_p%2.3f_fc%1.2f.pickle' % (p_val_threshold, fc_thresh)
    tf_per_path_file = '../clustering/tf_enrichment/tf_per_path_p%2.3f_fc%1.2f.pickle' % (p_val_threshold, fc_thresh)
    tf_mem_file = '../clustering/tf_enrichment/tf_per_mem_path_p%2.3f_fc%1.2f.pickle' % (p_val_threshold, fc_thresh)


    # tf_dict, gene_dict = ft.make_association_dict(association_file)
    # pd.to_pickle(gene_dict, association_dict_file)

    # Load previous pickles
    gene_dict = pd.read_pickle(association_dict_file)
    binary_df = pd.read_pickle(binary_path)     # Compressed strings
    path_dict = {'WT': {'binary_path': pd.read_pickle(wt_path)}, 'KO': {'binary_path': pd.read_pickle(ko_path)}}
    times = path_dict['WT']['binary_path'].columns.values
    steps = range(1, len(path_dict['WT']['binary_path'].columns.values))
    step_values = list(set(path_dict['WT']['binary_path'].values.astype(int).flatten()))
    string_to_array_dict, stepwise_path_combos = make_possible_paths(steps, step_values)
    # print np.cumsum(path_dict['WT']['binary_path'].values, axis=1)[:5]
    # sys.exit()
    plt.plot(np.cumsum(path_dict['KO']['binary_path'].drop_duplicates().values, axis=1).T, '-', c='b', alpha=0.1, lw=20)
    plt.show()
    sys.exit()

    growing_path = binary_to_growing_path(steps, path_dict['KO']['binary_path'])

    path_dictionary = find_genes_in_path(growing_path, stepwise_path_combos)

    sig_path_list = find_path_enrichment(path_dictionary, fdr=0.005)

    plot_enrichment_results(sig_path_list, string_to_array_dict)
