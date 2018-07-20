import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pydiffexp import DEAnalysis, DEPlot, DEResults, cluster_discrete
from pydiffexp.utils import all_subsets
from pydiffexp.utils import fisher_test as ft
from pipeline import DynamicDifferentialExpression as DDE
from pipeline import filter_dde
from palettable.cartocolors.qualitative import Bold_8, Prism_10
from goatools.obo_parser import GODag
from goatools import GOEnrichmentStudy
from scripts.go_enrichment import check_enrichment


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
    gene_dict = pd.read_pickle('../data/tf_associations/human_encode_associations.pkl')

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

    der = dea.results['{}'.format(contrast)]    # type: DEResults
    ts_der = dea.results['({})_ts'.format(contrast)]
    ar_der = dea.results['({})_ar'.format(contrast)]
    wt_ts_der = dea.results['{}_ts'.format(c_condition)]
    exp_ts_der = dea.results['{}_ts'.format(e_condition)]

    p = 0.05
    pairwise = set(der.top_table(p=p).index)

    ddegs = set(filter_dde(der.score_clustering()).index).intersection(pairwise)
    hm_data = der.top_table().loc[
        der.score_clustering().loc[pairwise].sort_values('Cluster', ascending=False).index]
    hm_data = hm_data.iloc[:, :len(dea.times)]
    hm_data.columns = dea.times

    # Differentially responding genes
    ar_dt = set(ar_der.top_table(p=p).index)

    # Unclear why, but the AR and TS trajectory genes match
    assert ar_dt == set(ts_der.top_table(p=p).index)
    drgs = ar_dt

    gene_sets = {'degs': pairwise, 'ddegs': ddegs, 'drgs': drgs}
    set_sizes, set_collections = all_subsets([pairwise, ddegs, drgs],
                                             ['degs', 'ddegs', 'drgs'])

    #
    obo = '../data/goa_data/go-basic.obo'
    # association = "../data/goa_data/human_go_associations.txt"
    #
    # assoc = read_associations(association)
    obo_dag = GODag(obo_file=obo)
    # pop = set(ensembl_to_hgnc.hgnc_symbol)
    # methods = ["bonferroni", "sidak", "holm"]
    # g = GOEnrichmentStudy(pop, assoc, obo_dag, alpha=0.05, methods=methods)

    # all_genes_tf, all_tf_dict = ft.convert_gene_to_tf(pop, gene_dict)
    # deg_tf, _ = ft.convert_gene_to_tf(ensembl_to_hgnc.loc[pairwise, 'hgnc_symbol'].values, gene_dict)

    go_sets = {}
    go_enrich = {}
    # tf_sets = {}
    for gene_class, gene_set in gene_sets.items():
        # Write the gene list to a file
        gene_set_path = 'GSE69822/go_enrich/pten_{}_list.txt'.format(gene_class)

        hgnc_genes = ensembl_to_hgnc.loc[gene_set, 'hgnc_symbol'].values

        # filtered_tf, filtered_tf_dict = ft.convert_gene_to_tf(hgnc_genes, gene_dict)
        # # if gene_class == 'drgs':
        # #     bg = deg_tf
        # # else:
        # #     bg = all_genes_tf
        # tf_enrich = ft.calculate_study_enrichment(filtered_tf, all_genes_tf)
        # tf_sets[gene_class] = set(tf_enrich[tf_enrich.p_bonferroni<0.05].TF)

        write_gene_list(hgnc_genes, gene_set_path)
        enrich_path = gene_set_path.replace('list', 'enrich')
        try:
            enrich = pd.read_csv(enrich_path, sep='\t')
        except FileNotFoundError:
            r = g.run_study(frozenset(hgnc_genes))
            g.wr_tsv(enrich_path, r)
            enrich = pd.read_csv(enrich_path, sep='\t')
        enrich = enrich[(enrich.p_bonferroni < 0.05)]
        go_sets[gene_class] = set(enrich['name'].values)
        go_enrich[gene_class] = set(enrich['# GO'].values)

    go_sizes, go_collections = all_subsets([go_sets['degs'], go_sets['ddegs'], go_sets['drgs']],
                                           ['degs', 'ddegs', 'drgs'])
    term_sizes, term_collections = all_subsets([go_enrich['degs'], go_enrich['ddegs'], go_enrich['drgs']],
                                           ['degs', 'ddegs', 'drgs'])

    # tf_sizes, tf_collections = all_subsets([tf_sets['degs'], tf_sets['ddegs'], tf_sets['drgs']],
    #                                        ['degs', 'ddegs', 'drgs'])
    # print(tf_sizes)

    # Next find the genes which have at least one identifiable difference

    # Different ar trajectories with identifiable points of change
    # Usually 'significant' individual p-values drop out with a correction
    ar_signs = ar_der.decide_tests(p=p).loc[drgs]
    ar_signs = ar_signs[(ar_signs != 0).any(axis=1)]
    ar_genes = set(ar_signs.index)

    # Different ar trajectories with identifiable points of change
    # Usually 'significant' individual p-values drop out with a correction
    ts_signs = ts_der.decide_tests(p=p).loc[drgs]
    ts_signs = ts_signs[(ts_signs != 0).any(axis=1)]
    ts_genes = set(ts_signs.index)


    # It is challenging to find slopes that are significantly nonzero
    # A more liberal approach is to mesh the discrete steps of the independent
    # TS trajectories
    ts_fraction = 0.1
    if len(ts_genes) < (1-ts_fraction)*len(drgs):
        pten_signs = exp_ts_der.discrete
        wt_signs = wt_ts_der.discrete
        pten_signs.columns = dea.times[1:]
        wt_signs.columns = dea.times[1:]
        diff_signs = np.sign(pten_signs - wt_signs)
    else:
        diff_signs = ts_signs

    ts_diff_signs = diff_signs.loc[drgs]
    ts_path_df = np.cumsum(np.sign(ts_diff_signs[(ts_diff_signs != 0).any(axis=1)]), axis=1)


    # Get barplot_data
    set_data = set_sizes.copy()
    term_depths = {k: [obo_dag.query_term(term).depth for term in v] for k, v in term_collections.items()}
    set_data['GO terms'] = go_sizes['size']
    set_data = (set_data / set_data.sum()).stack().reset_index()
    set_data.replace({'size': 'genes'}, inplace=True)
    set_data.columns = ['Gene Category', 'Unique #', "Fraction of Total"]

    # Organize plot
    # Overall gridspec with 1 row, two columns
    f = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(1, 2)

    # Create a gridspec within the gridspec. 1 row and 2 columns, specifying width ratio
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0],
                                               width_ratios=[len(dea.times), 2],
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
    # tf_ax = plt.subplot(gs_right[2])

    hash_data = pd.DataFrame(index=hm_data.index).astype(int)
    hash_data['ddeg'] = [ii in ddegs for ii in hash_data.index]
    hash_data['drg'] = [ii in drgs for ii in hash_data.index]
    hash_data = hash_data.astype(int)

    hm_ax, hash_ax = dep.heatmap(hm_data, hash_data, hm_ax=hm_ax, hash_ax=hash_ax,
                                 cbar_ax=cbar_ax, yticklabels=False,
                                 cbar_kws=dict(orientation='horizontal',
                                               ticks=[-1, 0, 1]))
    cbar_ax.xaxis.tick_top()
    cbar_ax.invert_xaxis()
    hidden_ax.set_xlabel('')
    hidden_ax.set_ylabel('')
    hidden_ax.axis('off')

    index_order = ['degs', 'ddegs', 'drgs', 'degs∩ddegs', 'degs∩drgs',
                   'ddegs∩drgs','degs∩ddegs∩drgs', 'all']

    y_order = list(reversed(range(len(set_sizes))))
    # gene_ax = sns.barplot(data=set_data, y='Gene Category', x='Fraction of Total',
    #                       hue='Unique #', log=True, ax=gene_ax, order=index_order)
    gene_ax.axis('off')
    gene_ax.set_ylabel('')
    # gene_ax.set_title(gene_ax.get_xlabel())
    # gene_ax.set_xlabel('')

    x = pd.DataFrame()
    for gc, terms in term_collections.items():
        for t in terms:
            x = x.append(pd.Series([gc, obo_dag.query_term(t).depth],
                                   index=['gene class', 'term depth'], name=t))

    c_index = [1, 7, 5, 9, 3, 6]
    colors = [Prism_10.mpl_colors[idx] for idx in c_index] + ['k', '0.5']
    cmap = {gc: colors[ii] for ii, gc in enumerate(index_order)}

    full_dist = x.copy()
    full_dist['gene class'] = "all"
    full_dist.reset_index(drop=True, inplace=True)
    all_x = pd.concat([x, full_dist])

    go_ax = sns.boxplot(data=all_x, x='term depth', y='gene class',
                        order=index_order, ax=go_ax,
                        showfliers=False, boxprops=dict(linewidth=0),
                        medianprops=dict(solid_capstyle='butt', color='w'),
                        palette=cmap)
    xx = x[[gc in term_sizes[term_sizes['size'] < 30].index for gc in x['gene class']]]
    go_ax = sns.swarmplot(data=xx, x='term depth', y='gene class', order=index_order, ax=go_ax, color='k')
    go_ax.plot([x['term depth'].median(), x['term depth'].median()],
               go_ax.get_ylim(), 'k-', lw=2, zorder=0, c='0.25')
    y_ticks = ["n={}".format(term_sizes.loc[gc, 'size'] if gc != 'all'
                             else sum(term_sizes['size'])) for gc in index_order]
    go_ax.set_yticklabels(y_ticks)
    go_ax.set_ylabel('')

    # tf_ax.barh(y=y_order, width=tf_sizes['size'], tick_label=tf_sizes['index'], log=True, color=cmap)
    # tf_ax.set_ylabel('')
    # tf_ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig('pten_gene_classification.pdf')
    sys.exit()

    # Set the type of plot to display
    path_df = ar_signs
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
    sys.exit()




