import os
import sys
import functools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import discretized_clustering as dcluster
from matplotlib.colors import ColorConverter
import matplotlib as mpl


def is_square(n):
    """
    Determine if a number is a perfect square
    :param n: int or float
        The number to check
    :return: Boolean
        Return True if the number is a perfect square
    """
    return np.sqrt(n).is_integer()


def get_factors(n):
    """
    Calculate the factors of a number
    :param n: int
        The number to be factored
    :return: list
        A sorted list of the unique factors from smallest to largest
    """
    factor_list = list(set(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))
    factor_list.sort()
    return factor_list


def calc_subplot_dimensions(x):
    """
    Calculate the dimensions for a matplotlib subplot object.
    :param x: int
        Number of plots that need to be made
    :return: rows, columns
        The number of rows and columns that should be in the subplot
    """
    if x <= 3:
        rows = x
        columns = 1
    else:
        factor_list = get_factors(x)
        while len(factor_list) <= 2 and not is_square(x):
            x += 1
            factor_list = get_factors(x)
        if is_square(x):
            rows = int(np.sqrt(x))
            columns = int(np.sqrt(x))

        else:
            rows = factor_list[len(factor_list)/2-1]
            columns = factor_list[len(factor_list)/2]

    return rows, columns

if __name__ == '__main__':

    # Getting data files.
    directory = '../'
    dea_path = directory + 'data/pickles/protocol_2_de_analysis_full_BGcorrected.pickle'
    dc_path = '../data/pickles/'
    r_results_path_diff = directory + 'clustering/expression_change_diff/'
    r_results_path = directory + 'clustering/expression_change/'
    fdr = 0.05
    p_thresh = 0.01
    fc_thresh = 0.2
    print(fdr, p_thresh, fc_thresh)
    save_fmt = 'pdf'
    pickle_string = (dc_path + 'discretized_clustering_results_p%s_fc%s_fdr%s.pickle'
                     % (str(p_thresh), str(fc_thresh), str(fdr)))

    # Set matplotlib font parameters
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.sans-serif'] = 'Arial'

    dc_diff = dcluster.DiscretizedClusterer(dea_path, r_results_path_diff, p_value_threshold=p_thresh,
                                            fold_change_threshold=fc_thresh)
    dc_diff.cluster_genes()
    r_diff = dc_diff.cluster_dict['diff']

    # If discretized clustering results exists don't rerun them.
    dc = dcluster.DiscretizedClusterer(dea_path, r_results_path, p_value_threshold=p_thresh,
                                       fold_change_threshold=fc_thresh)

    if not os.path.isfile(pickle_string):
        dc.cluster_genes()
        dc.get_enrichment(fdr=fdr)
        pd.to_pickle(dc, pickle_string)
    #
    #
    dc = pd.read_pickle(pickle_string)
    #
    # # Get differences
    # diff = dc.cluster_dict['WT']-dc.cluster_dict['KO']
    #
    # # Remove ones that don't change
    # diff = diff[(diff.T !=0).any()]
    # print diff
    # sys.exit()
    #
    # # Calculate cumulative differences
    # wt_cum = np.cumsum(dc.cluster_dict['WT'].loc[diff.index], axis=1)
    # ko_cum = np.cumsum(dc.cluster_dict['KO'].loc[diff.index], axis=1)
    # cum_diff = wt_cum-ko_cum
    #
    # # # Plot all flows for both cases.
    # sets = ['WT', 'KO']
    # colors = ['m', 'c']
    # alphas = [1.0, 0.5]
    # paths = ['all', 'all']
    #
    # fig, ax = plt.subplots(figsize=(10, 7.5))
    # dc.plot_flows(ax, ['diff'], colors, alphas, paths, x_coords=dc.times, min_sw=0.01, max_sw=1, uniform=False, path_df=diff)
    # plt.show()
    # sys.exit()
    #
    # dc.plot_flows(ax, sets, colors, alphas, paths, x_coords=dc.times, min_sw=0.01, max_sw=1, uniform=False)
    #
    # background = np.array((0.5, 0.5, 0.5))
    # cc = ColorConverter()
    # for c, s, a in zip(colors, sets, alphas):
    #     ax.plot([], [], '-', color=c, label=s, lw=6, alpha=a)
    #     background = (1 - a)*background + a * np.array(cc.to_rgb(c))
    # ax.plot([], [], '-', color=background, label='Merge', lw=6)
    # ax.legend(loc='best', fontsize=18)
    #
    # ax.set_xticks(dc.times)
    # ax.set_xlabel('Time (min)', fontsize=24, weight='bold')
    # ax.set_ylabel('Discretized Change', fontsize=24, weight='bold')
    # plt.tick_params(axis='both', which='major', labelsize=18)
    # save_name = 'discretized_clustering_gene_flow_p%s_fc%s_fdr%s.%s' % (str(p_thresh), str(fc_thresh), str(fdr), save_fmt)
    # plt.tight_layout()
    # # plt.savefig(save_name, format=save_fmt)
    # plt.show()
    # plt.close()
    # sys.exit()

    tick_labels = [0, 0, 15, 60, 120, 240]
    for condition in dc.conditions:
        results = dc.enrichment_results[condition]
        paths = ['.'.join(dc.string_to_array_dict[path][:].astype(str).tolist()) for path in results['significant_paths']]
        if len(paths) == 0:
            continue
        n_rows, n_cols = calc_subplot_dimensions(len(paths))
        f = plt.figure(figsize=(5,10))
        a, b = np.meshgrid(range(n_rows), range(n_cols))
        row_index = b.flatten()+1
        column_index = a.flatten()+1
        for ii, path in enumerate(paths):
            # if path =='0.0.0.0.-1':
            #     print path
            #     current_ax = f.add_subplot(1, 1, 1)
            # else:
            #     continue
            current_ax = f.add_subplot(n_rows, n_cols, ii+1)
            sets = [condition, condition]
            colors = ['0.75', 'g']
            alphas = [1.0, 1]
            flows = ['all', path]
            dc.plot_flows(current_ax, sets, colors, alphas, flows, min_sw=0.06, max_sw=0.75)
            for c, s, a in zip(colors, flows, alphas):
                current_ax.plot([], [], '-', color=c, label=s, lw=6, alpha=a)
            # ax.legend(loc='best')

            title = 'Path: ' + u'\u2192'.join(path.split('.'))
            print(path)
            print(dc.enrichment_results[condition]['enrichment_tables'][ii])
            print()
            current_ax.set_title(title, fontsize=24, weight='bold')
            current_ax.tick_params(axis='both', labelsize=24)
            save_str = path
        # plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.35, wspace=0.2)
        plt.tight_layout(w_pad=0.1)
        # plt.show()
        plt.savefig(condition+"_"+save_str + str(fdr) + '.pdf', format='pdf')
        plt.close()
