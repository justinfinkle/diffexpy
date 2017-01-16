__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import discretized_clustering as dcluster
from matplotlib.colors import ColorConverter
import matplotlib as mpl
import fisher_test as ft

if __name__ == '__main__':

    # Getting data files.
    directory = '../'
    dea_path = directory + 'data/pickles/protocol_2_de_analysis_full_BGcorrected.pickle'
    dc_path = '../data/pickles/'
    r_results_path_diff = directory + 'clustering/differential_expression/'
    r_results_path = directory + 'clustering/expression_change/'
    fdr = 0.05
    p_thresh = 0.001
    fc_thresh = 0.5
    save_fmt = 'pdf'
    pickle_string = (dc_path + 'discretized_clustering_results_p%s_fc%s_fdr%s.pickle'
                     % (str(p_thresh), str(fc_thresh), str(fdr)))

    # Set matplotlib font parameters
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['font.sans-serif'] = 'Arial'

    dc_diff = dcluster.DiscretizedClusterer(dea_path, r_results_path_diff, p_value_threshold=p_thresh,
                                            fold_change_threshold=fc_thresh)
    association_dict_file = '../clustering/tf_enrichment/gene_association_dict.pickle'
    association_dict = pd.read_pickle(association_dict_file)
    background_tfs = []
    for value in association_dict.itervalues():
        background_tfs += value

    for comp, limma in dc_diff.limma_results.iteritems():
        if '_0' in comp:
            continue
        print comp
        filtered = limma[(limma['adj.P.Val'] < p_thresh) & (np.abs(limma['logFC']) >= fc_thresh)]
        neg = filtered[filtered['logFC'] < 0]
        pos = filtered[filtered['logFC'] > 0]
        # print neg
        # sys.exit()
        genes = pos.index.values
        # background_tfs = ft.convert_gene_to_tf(set(limma.index.values).difference(set(genes)), association_dict)
        tfs = ft.convert_gene_to_tf(genes, association_dict)
        results = ft.calculate_study_enrichment(tfs, background_tfs)
        if len(results[results['FDR_reject']]) > 0:
            print results[results['FDR_reject']]
            # for g in genes:
            #     print g
            # print
            raw_input()
    sys.exit()
    dc_diff.cluster_genes()
    r_diff = dc_diff.cluster_dict['diff']
    # print r_diff
    # sys.exit()
    # dc_diff.get_enrichment(fdr=fdr)
    # sys.exit()

    # Compile fold change differences that are significant
    # raw_fc = pd.DataFrame()
    # for time in dc_diff.times:
    #     comp = [c for c in dc_diff.limma_comparisons if "_"+str(time) in c]
    #     fc = dc_diff.limma_results[comp[0]].copy()
    #     fc.loc[fc['B'] <= 4.6, 'logFC'] = 0
    #     raw_fc = pd.concat([raw_fc, fc['logFC']], join='outer', axis=1)
    # raw_fc = raw_fc[(raw_fc.T!=0).any()]
    # raw_fc.columns = dc_diff.times

    r_diff = dc_diff.cluster_dict['diff']
    # dc_diff.get_enrichment(fdr=fdr)
    # sys.exit()

    # If discretized clustering results exists don't rerun them.
    dc = dcluster.DiscretizedClusterer(dea_path, r_results_path, p_value_threshold=p_thresh,
                                       fold_change_threshold=fc_thresh)

    dc.cluster_genes()
    # for key, value in dc.limma_results.iteritems():
    #     if 'diff' not in key and 'KO' in key:
    #         print key, '\n'
    #         for gene in value[value.B > 4.6].index:
    #             print gene
    #         print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    #         print
    # sys.exit()
    combined_diff = pd.concat([r_diff, dc.cluster_dict['WT'], dc.cluster_dict['KO']], join='inner', axis=1)

    # dc.conditions = ['WT', 'KO']

    grouped = combined_diff.groupby(combined_diff.columns, axis=1)

    # for name, group in grouped:
    #     sig_diff = group[(group.iloc[:, 0] != 0)].astype(int).astype(str)
    #     if len(sig_diff) == 0:
    #         continue
    #     sig_diff.columns = ['D', 'WT', 'KO']
    #     sig_diff['combo'] = sig_diff['D'] + "_" + sig_diff['WT'] + "_" + sig_diff['KO']
    #     combos = list(set(sig_diff['combo'].values.tolist()))
    #     print "End time: ", name
    #     for combo in combos:
    #         genes = sig_diff[sig_diff.loc[:, 'combo'] == combo].index.values
    #         tfs = ft.convert_gene_to_tf(genes, association_dict)
    #         print combo, len(genes), len(tfs)
    #         results = ft.calculate_study_enrichment(tfs, background_tfs)
    #         if len(results[results['FDR_reject']]) > 0:
    #             print results[results['FDR_reject']]
    #             for g in genes:
    #                 print g
    #             print
    #             raw_input()
    #
    # sys.exit()
    # Plot differences
    wt = dc.cluster_dict['WT'].loc[combined_diff.index]
    ko = dc.cluster_dict['KO'].loc[combined_diff.index]
    diff = wt-ko

    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5))
    background = np.array((0.5, 0.5, 0.5))
    cc = ColorConverter()
    for c, a in zip(['m', 'c'], [1, 0.5]):
        background = (1 - a) * background + a * np.array(cc.to_rgb(c))

    sets = ['WT', 'KO', 'diff']
    colors = ['m', 'c', background]
    alphas = [1, 1, 1]
    paths = ['all']*3
    norm = len(diff)
    ii = 0
    for s, c, a, p in zip(sets, colors, alphas, paths):
        if s == 'diff':
            p_df = diff
        else:
            p_df = None
        cur_ax = ax[ii]
        dc.plot_flows(cur_ax, [s], [c], [a], [p], x_coords=dc.times, min_sw=0.01, max_sw=1, uniform=False,
                      path_df=p_df, genes=combined_diff.index.values, norm=norm)
        ii += 1
    plt.tight_layout()
    plt.show()
    sys.exit()


    sys.exit()