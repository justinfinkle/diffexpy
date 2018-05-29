import ast
import os
import shutil
import sys
import tarfile

import numpy as np
import pandas as pd
import seaborn as sns
from pydiffexp import DEAnalysis, DEResults, DEPlot, get_scores, cluster_discrete, pairwise_corr
from pydiffexp.gnw import mk_ch_dir, GnwNetResults, GnwSimResults, draw_results, get_graph
from scipy import stats
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import networkx as nx


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


def save_permutes(save_dir, grouped_data, n=100):
    """
    Save permutes to a directory as a pickles
    :param save_dir:
    :param grouped_data:
    :param n:

    :return:
    """
    for i in range(n):
        print("Saving permute {}".format(i))
        shuffled = grouped_data.apply(shuffle)   # type: pd.DataFrame
        shuffled.to_pickle("{}/{}_permuted.pkl".format(save_dir, i))
    return


def shuffle(df):
    new_array = df.values.copy()
    _ = [np.random.shuffle(i) for i in new_array]
    new_df = pd.DataFrame(new_array, index=df.index, columns=df.columns)
    return new_df


def analyze_permutes(real_scores, permutes_path, contrast) -> (pd.DataFrame, pd.DataFrame):
    p_score = pd.DataFrame()
    n_permutes = len(os.listdir(permutes_path))
    for p in os.listdir(permutes_path):
        print(p)
        # Load data and fit to get permuted data p-values
        p_idx = p.split("_")[0]
        p_data = pd.read_pickle(os.path.join(permutes_path, p))
        _, cur_scores = fit_dea(p_data, contrast, reference_labels=['condition', 'time'], log2=False)

        # drop cluster column, rename score, and add to real scores
        cur_scores.drop('Cluster', inplace=True, axis=1)
        cur_scores.columns = ['p{}_score'.format(p_idx)]
        p_score = pd.concat([p_score, cur_scores], axis=1)

    p_mean = p_score.mean(axis=1)
    p_std = p_score.std(axis=1)
    values = real_scores['score'].copy()
    p_z = -1 * ((values - p_mean) / p_std).abs()
    p_values = (2 * p_z.apply(stats.norm.cdf)).round(decimals=int(n_permutes/10))
    p_values.name = 'p_value'
    new_scores = pd.concat([real_scores, p_values], axis=1)         # type: pd.DataFrame

    return new_scores, p_score


def compress_directory(directory, source='.'):
    """
    Compress a directory to a tarball
    :param directory:
    :param source:
    :return:
    """
    with tarfile.open("{}.tar.gz".format(directory), 'w:gz') as tar:
        tar.add(directory, arcname=os.path.basename(source))
    tar.close()


def fit_dea(data, default_contrast, **kwargs):
    # Create and fit analysis object
    dea = DEAnalysis(data, **kwargs)
    dea.fit_contrasts(dea.default_contrasts[default_contrast]['contrasts'], fit_names=default_contrast)
    der = dea.results[default_contrast]  # type: DEResults
    scores = der.score_clustering()
    return dea, scores


def dde(data, default_contrast, project_dir, n_permutes=100, permute_path=None, save_permute_data=False, calc_p=False,
        compress=True, **kwargs):
    # Set defaults
    kwargs.setdefault('reference_labels', ['condition', 'time'])
    kwargs.setdefault('index_names', ['condition', 'replicate', 'time'])

    # Make the project directory to store the output
    project_path = os.path.abspath(project_dir)
    dir_name = os.path.split(project_path)[1]
    print("Saving project files to {}".format(project_path))
    mk_ch_dir(project_path, ch=False)
    prefix = "{}/{}_{}_".format(project_path, dir_name, default_contrast)

    dea, scores = fit_dea(data, default_contrast, **kwargs)
    dea.to_pickle("{}dea.pkl".format(prefix), force_save=True)

    # Make a directory for the permutes
    if permute_path is None:
        permute_path = "{}permutes".format(prefix)
    mk_ch_dir(permute_path, ch=False)

    if save_permute_data:
        print("Creating permute data")
        idx = pd.IndexSlice
        data = dea.data.loc[:, idx[default_contrast.split('-'), :, :]]
        grouped = data.groupby(level='condition', axis=1)
        save_permutes(permute_path, grouped, n=n_permutes)

    if calc_p and len(os.listdir(permute_path)):
        print("Calculating cluster ranking pvalues")
        scores, ptest_scores = analyze_permutes(scores, permute_path, default_contrast)
        print('Saving permutation test results')
        scores.to_pickle("{}dde.pkl".format(prefix))
        ptest_scores.to_pickle("{}ptest_scores.pkl".format(prefix))

    if compress:
        print('Compressing permute directory')
        compress_directory(permute_path)
        print('Removing permutes directory')
        shutil.rmtree(permute_path)

    return dea


def compile_sim(sim_dir, times, save_path=None, **kwargs):
    # Initialize the results object
    gnr = GnwNetResults(sim_dir, **kwargs)

    print("Compiling simulation results. This could take a while")
    sim_results = gnr.compile_results(censor_times=times, save_intermediates=False)
    if save_path is not None:
        sim_results.to_pickle(save_path)
    return sim_results


def filter_dde(df, col='Cluster', thresh=2):
    """
    Filter out dde genes that are "uninteresting". They have fewer unique discrete labels than the threshold
    e.g. thresh=2 and Cluster = (1,1,1,1) will have only 1 unique label, and will be filtered out.
    :param df:
    :param col:
    :param thresh:
    :return:
    """
    df = df.loc[df[col].apply(ast.literal_eval).apply(set).apply(len) >= thresh]
    return df


def discretize_sim(sim_data: pd.DataFrame, p=0.05, filter_interesting=True, fillna=True):

    if fillna:
        sim_data.fillna(0, inplace=True)

    weighted_lfc = ((1 - sim_data.loc[:, 'lfc_pvalue']) * sim_data.loc[:, 'lfc'])
    discrete = sim_data.loc[:, 'lfc'].apply(np.sign).astype(int)
    clusters = cluster_discrete((discrete * (sim_data.loc[:, 'lfc_pvalue'] < p)))
    g = clusters.groupby('Cluster')
    scores = get_scores(g, sim_data.loc[:, 'lfc'], weighted_lfc).sort_index()

    if filter_interesting:
        scores = filter_dde(scores)

    # Reorganize the index to match the input one
    scores.set_index(['x_perturbation', 'id'], append=True, inplace=True)
    scores = scores.swaplevel(i='id', j='gene').sort_index()

    return scores


def correlate(experimental: pd.DataFrame, simulated: pd.DataFrame, sim_conditions, sim_node=None):
    print('Computing pairwise')
    gene_mean = experimental.groupby(level=['condition', 'time'], axis=1).mean()
    gene_mean_grouped = gene_mean.groupby(level='condition', axis=1)
    mean_z = gene_mean_grouped.transform(stats.zscore, ddof=1).fillna(0)

    # Correlate zscored means for each gene with each node in every simulation
    if sim_node is not None:
        simulated = simulated[simulated.index.get_level_values('gene') == sim_node]
    sim_means = simulated.loc[:, sim_conditions]
    sim_mean_z = sim_means.groupby(level='stat', axis=1).transform(stats.zscore, ddof=1).fillna(0)

    pcorr, p = pairwise_corr(sim_mean_z, mean_z, axis=1)
    print('Done')
    return pcorr, p


def display_sim(network, perturbation, times, directory, exp_condition='ko', ctrl_condition='wt'):
    data_dir = '{}/{}/'.format(directory, network)
    network_structure = "{}{}_goldstandard_signed.tsv".format(data_dir, network)

    ctrl_gsr = GnwSimResults(data_dir, network, ctrl_condition, sim_suffix='dream4_timeseries.tsv',
                             perturb_suffix="dream4_timeseries_perturbations.tsv")
    exp_gsr = GnwSimResults(data_dir, network, exp_condition, sim_suffix='dream4_timeseries.tsv',
                            perturb_suffix="dream4_timeseries_perturbations.tsv")

    data = pd.concat([ctrl_gsr.data, exp_gsr.data]).T
    dg = get_graph(network_structure)
    titles = ["x", "y", "PI3K"]
    mapping = {'G': "PI3k"}
    dg = nx.relabel_nodes(dg, mapping)
    draw_results(data, perturbation, titles, times=times, g=dg)
    plt.tight_layout()


def match_to_gene(x, y, correlation, corrp):
    matching_results = pd.DataFrame()
    for gene, row in x.iterrows():
        candidate_nets = y.loc[y.Cluster == row.Cluster]
        cur_corr = correlation.loc[gene, candidate_nets.index]
        cur_p = corrp.loc[gene, candidate_nets.index]
        cur_corr.name = 'pearson_r'
        cur_p.name = 'pearson_p'
        ranking = pd.concat([candidate_nets, cur_corr, cur_p], axis=1)
        ranking['mean'] = (ranking['score'] + ranking['pearson_r']) / 2
        ranking = ranking.loc[ranking.index.get_level_values(2) == 'y']
        ranking['true_gene'] = gene
        matching_results = pd.concat([matching_results, ranking.reset_index()], ignore_index=True, join='inner')

    return matching_results


def clustering_hamming(x, y):
    ham = [hamming(cluster, y[ii]) for ii, cluster in enumerate(x)]
    return ham


def display_gene():
    pass


if __name__ == '__main__':
    pd.set_option('display.width', 250)

    # Prep the raw data
    project_name = "GSE69822"
    contrast = 'ko-wt'
    prefix = "{}/{}_{}_".format(project_name, project_name, contrast)
    times = [0, 15, 40, 90, 180, 300]
    raw = load_data('../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt')
    gene_map = pd.read_csv('../data/GSE69822/mcf10a_gene_names.csv', index_col=0)
    mk_ch_dir(project_name, ch=False)

    # Fit the data using a DEAnalysis object
    # dea = dde(raw, contrast, project_name, save_permute_data=False, calc_p=False, voom=True)
    scores = pd.read_pickle("{}dde.pkl".format(prefix))
    dea = pd.read_pickle("{}dea.pkl".format(prefix))            # type: DEAnalysis
    plt.plot(dea.data.mean(axis=1), dea.data.std(axis=1), '.')
    plt.show()
    sys.exit()
    der = dea.results[contrast]                                 # type: DEResults

    # Compile simulation results
    # sim_stats = compile_sim('../data/motif_library/gnw_networks/', times=times,
    #                         save_path="{}sim_stats.pkl".format(prefix))
    sim_stats = pd.read_pickle("{}sim_stats.pkl".format(prefix))    # type: pd.DataFrame

    # Discretize the simulation stats and cluster
    sim_scores = discretize_sim(sim_stats)

    # Filter out "uninteresting" genes.
    dde_genes = filter_dde(scores)
    dde_genes = dde_genes[dde_genes['p_value'] < 0.05].sort_values('score', ascending=False)
    filtered_data = dea.voom_data.loc[dde_genes.index, contrast.split('-')]

    # Heatmap of expression

    # de_data = (der.top_table().iloc[:, :6])  # .multiply(der.p_value < p_thresh)
    # sort_idx = dde_genes.sort_values(['Cluster', 'score'], ascending=False).index.values
    # hm_data = de_data.loc[sort_idx]
    # hm_data = stats.zscore(hm_data, ddof=1, axis=1)
    # # hm_data = hm_data.divide(hm_data.abs().max(axis=1), axis=0)
    #
    # cmap = sns.diverging_palette(30, 260, s=80, l=55, as_cmap=True)
    # plt.figure(figsize=(4, 8))
    # sns.heatmap(hm_data, xticklabels=dea.times, yticklabels=False, cmap=cmap)
    # plt.xticks(rotation=45)
    # plt.title('PI3K KO DDE')
    # plt.tight_layout()
    # plt.show()
    # sys.exit()

    # Correlate genes with simulations
    corr, p = correlate(filtered_data, sim_stats.loc[sim_scores.index], ['ko_mean', 'wt_mean'], sim_node='y')

    # Combine score and correlations
    match = match_to_gene(dde_genes, sim_scores, corr, p)
    match.set_index(['id', 'x_perturbation', 'gene'], inplace=True)
    matched_genes = list(set(match.true_gene))
    unique_nets = list(set(match.index))

    # Need to validate
    true_prefix = prefix.replace('ko', 'ki')
    contrast = contrast.replace('ko', 'ki')

    # Compile simulation results
    # sim_stats = compile_sim('../data/motif_library/gnw_networks/', times=times,
    #                         save_path="{}sim_stats.pkl".format(true_prefix), experimental='ki')

    # Calculate expected clusters
    test_sim = pd.read_pickle("{}sim_stats.pkl".format(true_prefix))  # type: pd.DataFrame
    test_scores = discretize_sim(test_sim, filter_interesting=False)
    test_scores = test_scores.loc[match.index]
    test_scores.columns = ['predicted_cluster', 'predicted_score']

    true_scores = pd.read_pickle("{}dde.pkl".format(true_prefix))
    true_dea = pd.read_pickle("{}dea.pkl".format(true_prefix))  # type: DEAnalysis
    true_der = true_dea.results[contrast]  # type: DEResults

    true_dde_genes = filter_dde(true_scores)
    true_dde_genes = true_dde_genes[true_dde_genes['p_value'] < 0.05].sort_values('score', ascending=False)
    true_filtered_data = true_dea.data.loc[true_dde_genes.index, contrast.split('-')]

    combined = pd.concat([match, test_scores], join='inner', axis=1)
    matched_scores = true_scores.loc[match.true_gene].reset_index(drop=True)
    matched_scores.columns = ['true_cluster', 'true_score', 'true_p_value']
    all = pd.concat([combined.reset_index(), matched_scores], axis=1)
    new_corr, new_p = correlate(true_dea.data.loc[matched_genes, 'ki'], test_sim.loc[unique_nets], ['ki_mean'])
    unstacked = new_corr.unstack().sort_index()
    up = new_p.unstack().sort_index()
    unstacked.name = 'pred_true_pearsonr'
    up.name = 'pred_true_pearsonr_p'
    all.set_index(['id', 'x_perturbation', 'gene', 'true_gene'], inplace=True)
    all = pd.concat([all, unstacked.loc[all.index], up.loc[all.index]], axis=1)    # type: pd.DataFrame

    # Ignore NaNs for now
    # todo: figure out where there are nans
    all = all[~all.isnull().any(axis=1)]
    all['hamming'] = clustering_hamming(all['predicted_cluster'].apply(ast.literal_eval),
                                        all['true_cluster'].apply(ast.literal_eval))

    # all = filter_dde(all, 'predicted_cluster')
    all['true_mean'] = (all['true_score'] + all['pred_true_pearsonr'])/2
    all = all[(all['mean'] > 0) & (all['true_p_value'] < 0.05) & (all['true_mean'] > 0) & (all['predicted_score'] > 0)]

    g = all.groupby(level=3)

    idx = g['mean'].transform(max) == all['mean']
    test = all[idx]
    print(test)
    sns.regplot('mean', 'true_mean', data=test)
    print(stats.pearsonr(test['mean'], test['true_mean']))
    plt.tight_layout()
    plt.show()
    sys.exit()
    all.sort_values(['mean', 'hamming', 'predicted_score', 'pred_true_pearsonr'], ascending=[False, True, False, False],
                    inplace=True)

    print(all)
    print(all.mean())
    plt.hist(all.pred_true_pearsonr, bins=30)
    plt.figure()
    plt.plot(all['mean'], all['pred_true_pearsonr'], '.')
    print(stats.pearsonr(all['mean'], all['pred_true_pearsonr']))
    plt.figure()
    plt.plot(all['mean'], all['hamming'], '.')

    plt.figure()
    plt.plot(all['mean'], all['true_mean'], '.')
    print(stats.pearsonr(all['mean'], all['true_mean']))
    plt.show()

    sys.exit()


    # grouped = match.groupby('true_gene')
    # a = 0
    # unique = 0
    # for gene, data in grouped:
    #     unique += 1
    #     if gene in true_dde_genes.index:
    #         print(data)
    #         a += 1
    # print(a)
    # print(true_dde_genes.shape)
    # print(unique)
    # sys.exit()
    # print(all)

    dep = DEPlot()
    dep.tsplot(dea.voom_data.loc['ENSG00000004799', contrast.replace('ki', 'ko').split('-')], legend=False)
    plt.tight_layout()

    dep.tsplot(true_dea.voom_data.loc['ENSG00000004799', contrast.split('-')], legend=False)
    plt.tight_layout()

    # Display results
    display_sim(1995, -1, times, "../data/motif_library/gnw_networks/")
    display_sim(1995, -1, times, "../data/motif_library/gnw_networks/", exp_condition='ki')
    plt.show()
