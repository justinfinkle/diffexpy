import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pydiffexp import DEAnalysis, DEPlot, DEResults, cluster_discrete
from pipeline import DynamicDifferentialExpression as DDE
from palettable.cartocolors.qualitative import Bold_8


def load_data(path, bg_shift=True, **kwargs):
    """

    :param path:
    :param bg_shift:
    :param kwargs:
    :return:
    """
    kwargs.setdefault('sep', ',')
    kwargs.setdefault('index_col', 0)

    # Load original data
    raw_data = pd.read_csv(path, **kwargs)

    if bg_shift:
        raw_data[raw_data <= 0] = 1
    return raw_data


if __name__ == '__main__':
    # Set globals
    sns.set_palette(Bold_8.mpl_colors)
    pd.set_option('display.width', 250)

    plot_mean_variance = False

    """
    ===================================
    ========== Load the data ==========
    ===================================
    """
    idx = pd.IndexSlice
    # Load sim data and cleanup appropriately
    sim_data = pd.read_pickle('../data/motif_library/gnw_networks/all_sim_compiled_for_gse69822.pkl')
    sim_data = sim_data.loc['y', idx[:, :, 1, :]]
    sim_data.columns = sim_data.columns.remove_unused_levels()
    sim_data.columns.set_names(['replicate', 'time'], level=[1, 3], inplace=True)
    sim_data.columns.set_levels(['ki', 'pten', 'wt'], level=0, inplace=True)

    # Prep the raw data
    project_name = "GSE69822"
    t = [0, 15, 40, 90, 180, 300]
    raw = load_data('../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt')
    ensembl_to_hgnc = pd.read_csv('../data/GSE69822/mcf10a_gene_names.csv', index_col=0)
    hgnc_to_ensembl = ensembl_to_hgnc.reset_index().set_index('hgnc_symbol')

    # Labels that can be used when making DE contrasts used by limma. This helps with setting defaults
    contrast_labels = ['condition', 'time']

    # Features of the samples taken that are used in calculating statistics
    sample_features = ['condition', 'replicate', 'time']
    raw_dea = DEAnalysis(raw, reference_labels=contrast_labels, index_names=sample_features)

    # sim_dea = DEAnalysis(sim_data, reference_labels=contrast_labels, index_names=sample_features)

    """
        ===================================
        ============= Training ============
        ===================================
    """
    e_condition = 'pten'  # The experimental condition used
    c_condition = 'wt'  # The control condition used

    # Remove unnecessary data
    basic_data = raw_dea.raw.loc[:, [e_condition, c_condition]]
    contrast = "{}-{}".format(e_condition, c_condition)
    # dde = DDE(project_name)

    # override = False    # Rerun certain parts of the analysis
    #
    # matches = dde.train(basic_data, project_name, sim_dea, experimental=e_condition,
    #                     counts=True, override=override)

    # dea = DEAnalysis(basic_data, reference_labels=contrast_labels, index_names=sample_features,
    #                  counts=True, log2=False)
    # dea.fit_contrasts(dea.default_contrasts)
    # dea.to_pickle('GSE69822/GSE69822_ki-wt_dea.pkl')
    dea = pd.read_pickle('GSE69822/GSE69822_{}_dea.pkl'.format(contrast))
    dep = DEPlot()

    der = dea.results['{}'.format(contrast)]
    ts_der = dea.results['({})_ts'.format(contrast)]
    ar_der = dea.results['({})_ar'.format(contrast)]
    wt_ts_der = dea.results['{}_ts'.format(c_condition)]
    exp_ts_der = dea.results['{}_ts'.format(e_condition)]

    p = 0.05
    pairwise = set(der.top_table(p=p).index)
    dde_genes = set(dde.dde_genes.index)
    ar_signs = ar_der.decide_tests()
    ar_genes = set(ar_signs[(ar_signs != 0).any(axis=1)].index)
    ts_signs = ts_der.decide_tests()
    ts_genes = set(ts_signs[(ts_signs != 0).any(axis=1)].index)

    ar_signs = dea.results['({})_ar'.format(contrast)].discrete
    ar_path_df = ar_signs[(ar_signs != 0).any(axis=1)]

    pten_signs = exp_ts_der.discrete
    wt_signs = wt_ts_der.discrete
    pten_signs.columns = dea.times[1:]
    wt_signs.columns = dea.times[1:]

    diff_signs = np.sign(pten_signs - wt_signs)

    ts_diff_signs = diff_signs.loc[ts_der.top_table(p=0.05).index]
    ts_path_df = np.cumsum(np.sign(ts_diff_signs[(ts_diff_signs != 0).any(axis=1)]), axis=1)

    ts_path_df.insert(0, 0, 0)
    ts_path_df.columns = [0, 15, 40, 90, 180, 300]
    print(ts_path_df.apply(pd.Series.value_counts, axis=0).fillna(0).sort_index(ascending=False).astype(int))
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.85])
    dep.plot_flows(ax, ['diff'], [Bold_8.mpl_colors[0]], [1], ['all'],
                   x_coords=ts_path_df.columns, min_sw=0.01, max_sw=1,
                   uniform=False, path_df=ts_path_df, node_width=10)
    plt.xlabel('Time (min)')
    plt.ylabel('Cumulative Trajectory Differences')
    plt.tight_layout()
    plt.show()


