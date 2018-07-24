import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pydiffexp import DEAnalysis, DEPlot, DEResults
from pydiffexp.utils import all_subsets
from pydiffexp.utils import multiindex_helpers as mi
from pydiffexp.utils import fisher_test as ft
from pipeline import filter_dde
from palettable.cartocolors.qualitative import Bold_8, Prism_10
from goatools.obo_parser import GODag
from goatools import GOEnrichmentStudy
import r2py_helpers as rh
import rpy2.robjects as robj
from rpy2.robjects.packages import importr

# Import discrete goodness of fit
dgof = importr('dgof')


def load_data(path, mi_level_names, bg_shift=True,**kwargs):
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

    df = mi.make_multiindex(raw_data, index_names=mi_level_names)

    return df


def write_gene_list(genes, path):
    with open(path, 'w') as gene_file:
        for g in genes:
            gene_file.write(g+'\n')

    return


def read_associations(assoc_fn):
    assoc = {}
    for row in open(assoc_fn):
        atoms = row.split()
        if len(atoms) == 2:
            a, b = atoms
        elif len(atoms) > 2 and row.count('\t') == 1:
            a, b = row.split("\t")
        else:
            continue
        b = set(b.split(";"))
        assoc[a] = b

    return assoc


def d_ks_test(x, y, **kwargs):
    """
    Discrete ks test from R package dgof
    :param x:
    :param y:
    :param kwargs:
    :return:
    """
    kwargs.setdefault('simulate_p_value', True)
    kwargs.setdefault('alternative', 'less')

    # Cast to integer first so r doesn't crash
    x = np.array(x).astype(int)
    y = np.array(y).astype(int)

    x = robj.IntVector(x)
    y = robj.IntVector(y)
    results = dgof.ks_test(x, y, **kwargs)
    results = rh.rvect_to_py(results)
    return results['p_value']


def plot_cluster_term_figure():
    pass


def load_sim_data(compiled_path):
    # todo: needs to be fully functionalized
    """
    Load simulation data to match
    :param compiled_path:
    :return:
    """
    idx = pd.IndexSlice
    # Load sim data and cleanup appropriately
    sim_data = pd.read_pickle(compiled_path)
    sim_data = sim_data.loc['y', idx[:, :, 1, :]]
    sim_data.columns = sim_data.columns.remove_unused_levels()
    sim_data.columns.set_names(['replicate', 'time'], level=[1, 3], inplace=True)
    sim_data.columns.set_levels(['ki', 'pten', 'wt'], level=0, inplace=True)

    sim_dea = DEAnalysis(sim_data, reference_labels=contrast_labels, index_names=sample_features)
    return sim_dea


def fit_dea(path, data=None, override=False, **dea_kwargs):
    """

    :param path:
    :param override:
    :return:
    """
    dea_kwargs.setdefault('counts', True)
    dea_kwargs.setdefault('log2', False)

    try:
        if override:
            raise ValueError('Override to retrain')
        dea = pd.read_pickle(path)
    except (FileNotFoundError, ValueError) as e:
        dea = DEAnalysis(data, **dea_kwargs)
        dea.fit_contrasts(dea.default_contrasts)
        dea.to_pickle(path)
    return dea


def get_gene_classes(dea, contrast, p=0.05, strict_dde=True):

    der = dea.results['{}'.format(contrast)]            # type: DEResults
    ts_der = dea.results['({})_ts'.format(contrast)]    # type: DEResults
    ar_der = dea.results['({})_ar'.format(contrast)]    # type: DEResults

    deg = set(der.top_table(p=p).index)
    dde = set(filter_dde(der.score_clustering()).index)
    if strict_dde:
        dde = dde.intersection(deg)

    # Differentially responding genes
    ar_dt = set(ar_der.top_table(p=p).index)

    # Unclear why, but the AR and TS trajectory genes match
    assert ar_dt == set(ts_der.top_table(p=p).index)
    drg = ar_dt

    # Maintain a sort order for pretty plots downstream
    gene_classes = OrderedDict([('DEG', deg), ('DDE', dde), ('DRG', drg)])

    return der, ar_der, ts_der, gene_classes


def sign_diff(dea, ts_der, genes, exp, ctrl, p=0.05, reduce=True):
    """
    Find genes with a slope difference
    :param dea:
    :param genes:
    :param exp:
    :param ctrl:
    :return:
    """
    wt_ts_der = dea.results['{}_ts'.format(ctrl)]       # type: DEResults
    exp_ts_der = dea.results['{}_ts'.format(exp)]       # type: DEResults

    # Different ar trajectories with identifiable points of change
    # Usually 'significant' individual p-values drop out with a correction
    ts_signs = ts_der.decide_tests(p=p).loc[genes]
    ts_signs = ts_signs[(ts_signs != 0).any(axis=1)]
    ts_genes = set(ts_signs.index)

    # It is challenging to find slopes that are significantly nonzero
    # A more liberal approach is to mesh the discrete steps of the independent
    # TS trajectories
    ts_fraction = 0.1
    if len(ts_genes) < (1-ts_fraction)*len(genes):
        pten_signs = exp_ts_der.discrete
        wt_signs = wt_ts_der.discrete
        pten_signs.columns = dea.times[1:]
        wt_signs.columns = dea.times[1:]
        diff_signs = np.sign(pten_signs - wt_signs)
    else:
        diff_signs = ts_signs

    if reduce:
        diff_signs = diff_signs.loc[genes].copy()

    return diff_signs


def get_heatmap_data(dea, der, genes, row_norm='max'):
    clusters = der.score_clustering()
    cluster_ordered_genes = clusters.loc[genes].sort_values('Cluster', ascending=False)
    df = der.top_table().loc[cluster_ordered_genes.index]
    df = df.iloc[:, :len(dea.times)]
    df.columns = dea.times

    if row_norm == 'max':
        df = df.divide(df.abs().max(axis=1), axis=0)
    elif row_norm == 'zscore':
        df = stats.zscore(df, ddof=1, axis=1)

    return df


def get_hash_data(df, gene_set_dict):
    hash_data = pd.DataFrame(index=df.index).astype(int)
    for gene_class, gene_set in gene_set_dict.items():
        hash_data[gene_class] = [ii in gene_set for ii in hash_data.index]

    hash_data = hash_data.astype(int)
    return hash_data


def term_enrichment(pop_genes, gene_sets, obo_path, assoc_path, folder,
                    condition, regenerate=False, test_sig=True, **kwargs):
    kwargs.setdefault('alpha', 0.05)
    kwargs.setdefault('methods', ["bonferroni", "sidak", "holm"])

    # Setup goatools enrichment
    if regenerate:
        assoc = read_associations(assoc_path)
        go_dag = GODag(obo_file=obo_path)
        pop = set(pop_genes)
        g = GOEnrichmentStudy(pop, assoc, go_dag, **kwargs)

    # go_enrich = OrderedDict()
    go_enrich = OrderedDict()

    for gc, genes in gene_sets.items():
        # Write the gene list to a file
        out_path = '{}/go_enrich/{}_{}_list.txt'.format(folder, condition, gc)

        write_gene_list(genes, out_path)
        enrich_path = out_path.replace('list', 'enrich')
        try:
            if regenerate:
                raise ValueError('Override to retrain')
            enrich = pd.read_csv(enrich_path, sep='\t', index_col=0)
        except (FileNotFoundError, ValueError) as e:
            r = g.run_study(frozenset(genes))
            g.wr_tsv(enrich_path, r)
            enrich = pd.read_csv(enrich_path, sep='\t', index_col=0)
        enrich = enrich[(enrich.p_bonferroni < kwargs['alpha'])]
        go_enrich[gc] = enrich

    # Compile the results
    # enrich_df = pd.concat(enrich_df, keys=gene_sets.keys())

    # Get the sets
    # go = enrich_df.groupby(level=0).apply(lambda x: set(x.index.get_level_values(1)))

    go_sizes, go_terms = all_subsets(go_enrich)
    all_terms = pd.concat(go_terms.values())
    all_depths = all_terms['depth']
    all_median = np.median(all_depths)

    if test_sig:
        for gene_class, terms in go_terms.items():
            d = terms['depth'].values
            if len(d) < 3:
                print(gene_class, ' Skipped')
                continue

            t_med = np.median(d)
            if t_med > all_median:
                alternative = 'less'
            elif t_med < all_median:
                alternative = 'greater'
            else:
                alternative = 'two.sided'
            ks_p = d_ks_test(d, all_depths, alternative=alternative)
            print(gene_class, t_med, all_median, ks_p, sep='\t')

    return pd.concat(go_terms.values(), keys=go_terms.keys())


def plot_collections(hm_data, hash_data, term_data, output='show'):
    # Organize plot
    # Overall gridspec with 1 row, two columns
    f = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(1, 2)

    # Create a gridspec within the gridspec. 1 row and 2 columns, specifying width ratio
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0],
                                               width_ratios=[hm_data.shape[1], 2],
                                               height_ratios=[1, 50],
                                               wspace=0.05, hspace=0.05)
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1],
                                                height_ratios=[1, 1.5],
                                                hspace=0.25)

    cbar_ax = plt.subplot(gs_left[0, 0])
    hidden_ax = plt.subplot(gs_left[0, 1])
    hm_ax = plt.subplot(gs_left[1, 0])
    hash_ax = plt.subplot(gs_left[1, 1])
    gene_ax = plt.subplot(gs_right[0])
    go_ax = plt.subplot(gs_right[1])

    # Hide the top right axes where the venn diagram goes
    gene_ax.axis('off')

    # Initialize plotter
    dep = DEPlot()

    hm_ax, hash_ax = dep.heatmap(hm_data, hash_data, hm_ax=hm_ax, hash_ax=hash_ax,
                                 cbar_ax=cbar_ax, yticklabels=False,
                                 cbar_kws=dict(orientation='horizontal',
                                               ticks=[-1, 0, 1]))
    cbar_ax.xaxis.tick_top()
    cbar_ax.invert_xaxis()
    hidden_ax.set_xlabel('')
    hidden_ax.set_ylabel('')
    hidden_ax.axis('off')

    index_order = ['DEG', 'DDE', 'DRG', 'DEG∩DDE', 'DEG∩DRG',
                   'DDE∩DRG', 'DEG∩DDE∩DRG', 'All']

    c_index = [1, 7, 5, 9, 3, 6]
    colors = [Prism_10.mpl_colors[idx] for idx in c_index] + ['k', '0.5']
    cmap = {gc: colors[ii] for ii, gc in enumerate(index_order)}

    x = term_data.reset_index(level=0)
    x.columns = ['gene_class'] + x.columns[1:].values.tolist()
    full_dist = x.copy()
    full_dist['gene_class'] = "All"
    full_dist.reset_index(drop=True, inplace=True)
    all_x = pd.concat([x, full_dist])
    g = all_x.groupby('gene_class')

    go_ax = sns.boxplot(data=g.filter(lambda xx: True), x='depth', y='gene_class',
                        order=index_order, ax=go_ax,
                        showfliers=False, boxprops=dict(linewidth=0),
                        medianprops=dict(solid_capstyle='butt', color='w'),
                        palette=cmap)

    small_groups = g.filter(lambda x: len(x) < 50)
    go_ax = sns.swarmplot(data=small_groups, x='depth', y='gene_class',
                          order=index_order, ax=go_ax, color='k')

    go_ax.plot([x['depth'].median(), x['depth'].median()],
               go_ax.get_ylim(), 'k-', lw=2, zorder=0, c='0.25')

    term_sizes = g.apply(len).reindex(index_order).fillna(0).astype(int)
    y_ticks = ["n={}".format(term_sizes.loc[idx]) for idx in index_order]
    go_ax.set_yticklabels(y_ticks)
    go_ax.set_ylabel('')
    plt.tight_layout()

    if output.lower() == 'show':
        plt.show()

    else:
        plt.savefig(output, fmt='pdf')


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
    # Prep the raw data
    project_name = "GSE69822"
    obo_file = '../data/goa_data/go-basic.obo'
    associations =  '../data/goa_data/human_go_associations.txt'
    t = [0, 15, 40, 90, 180, 300]
    data_path = '../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt'
    sim_data_path = '../data/motif_library/gnw_networks/all_sim_compiled_for_gse69822.pkl'

    # Labels that can be used when making DE contrasts used by limma. This helps with setting defaults
    contrast_labels = ['condition', 'time']

    # Features of the samples taken that are used in calculating statistics
    sample_features = ['condition', 'replicate', 'time']

    # Load the data
    raw = load_data(data_path, sample_features, bg_shift=False)

    ensembl_to_hgnc = pd.read_csv('../data/GSE69822/mcf10a_gene_names.csv', index_col=0)
    hgnc_to_ensembl = ensembl_to_hgnc.reset_index().set_index('hgnc_symbol')
    gene_dict = pd.read_pickle('../data/tf_associations/human_encode_associations.pkl')

    """
        ===================================
        ============= Training ============
        ===================================
    """
    collection_plots = True
    sankey_plots = False
    e_condition = ['ko', 'ki', 'pten']  # The experimental condition used
    c_condition = 'wt'  # The control condition used

    # Remove unnecessary data
    for e in e_condition:
        basic_data = raw.loc[:, [e, c_condition]]
        contrast = "{}-{}".format(e, c_condition)
        dea_path = '{}/{}_{}_dea.pkl'.format(project_name, project_name, contrast)

        dea = fit_dea(dea_path, reference_labels=contrast_labels, index_names=sample_features)
        der, ar_der, ts_der, gc = get_gene_classes(dea, contrast)
        print(e, len(gc['DDE']))

        if collection_plots:
            # Convert the ensembl symbols to hgnc for GO enrichment
            hgnc_set = OrderedDict([(k, set(ensembl_to_hgnc.loc[v, 'hgnc_symbol'].values))
                                    for k, v in gc.items()])
            enriched = term_enrichment(set(hgnc_to_ensembl.index), hgnc_set, obo_file,
                                       associations, project_name, e,
                                       test_sig=True, regenerate=False)

            # Get the data to plot
            hm_data = get_heatmap_data(dea, der, gc['DEG'])
            hash_keys = ['DDE', 'DRG']
            hash_data = get_hash_data(hm_data, {k: gc[k] for k in hash_keys})

            # Plot the data
            plot_collections(hm_data, hash_data, enriched)

        if sankey_plots:
            dep = DEPlot()
            all_genes_tf, all_tf_dict = ft.convert_gene_to_tf(set(hgnc_to_ensembl.index), gene_dict)

            filtered_ensmbl = ar_der.discrete[ar_der.discrete.iloc[:, 2] == 1].index
            filtered_genes = ensembl_to_hgnc.loc[filtered_ensmbl, 'hgnc_symbol']
            filtered_tf, filtered_tf_dict = ft.convert_gene_to_tf(filtered_genes, gene_dict)
            enrich = ft.calculate_study_enrichment(filtered_tf, all_genes_tf)
            print(enrich.FDR_reject.sum())
            print(enrich.head())
            sys.exit()

            # ts_diff_signs = sign_diff(dea, ts_der, gc['DRG'], e_condition, c_condition)
            # ts_path_df = np.cumsum(np.sign(ts_diff_signs[(ts_diff_signs != 0).any(axis=1)]), axis=1)
            clusters = []
            for kk in range(ar_der.discrete.shape[1]-1):
                c = ['1' if jj > kk else '0' for jj in range(ar_der.discrete.shape[1])]
                clusters.append('({})'.format(', '.join(c)))
            print(clusters)
            # Set the type of plot to display
            path_df = ar_der.discrete[(ar_der.discrete != 0).any(axis=1)]
            path_df.insert(0, 0, 0)
            path_df.columns = dea.times
            print(path_df.apply(pd.Series.value_counts, axis=0).fillna(0).sort_index(ascending=False).astype(int))
            fig = plt.figure(figsize=(10, 7.5))
            ax = fig.add_axes([0.1, 0.1, 0.6, 0.85])
            dep.plot_flows(ax, ['diff'], [Bold_8.mpl_colors[0]], [1], ['all'],
                           x_coords=path_df.columns, min_sw=0.01, max_sw=1,
                           uniform=False, path_df=path_df, node_width=10,
                           legend=False)
            plt.xlabel('Time (min)')
            plt.ylabel('Cumulative Trajectory Differences')
            plt.tight_layout()
            plt.show()






