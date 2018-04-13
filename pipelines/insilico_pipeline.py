import ast
import os
import shutil
import sys
import tarfile
import itertools as it
import multiprocessing as mp

import numpy as np
import pandas as pd
import seaborn as sns
from pydiffexp import DEAnalysis, DEResults, DEPlot, get_scores, cluster_discrete, pairwise_corr
from pydiffexp.gnw import mk_ch_dir, GnwNetResults, GnwSimResults, draw_results, get_graph, tsv_to_dg
from scipy import stats
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
import networkx as nx


def load_insilico_data(path, conditions, stimuli, net_name, times=None):
    # List comprehension: for each combo of stimuli and conditions make a GSR object and get the data
    data = [GnwSimResults('{}/{}/'.format(path, ss), net_name, cc, sim_suffix='dream4_timeseries.tsv',
                          perturb_suffix="dream4_timeseries_perturbations.tsv", censor_times=times).data
            for ss, cc in it.product(stimuli, conditions)]

    # Concatenate into one dataframe
    data = pd.concat(data)

    return data

    # Old code
    data = pd.DataFrame()
    for stim in ['activating', 'deactivating']:
        for c in conditions:
            ts_file = '{bd}/{s}/{c}_sim_anon/Yeast-100_anon_{c}_dream4_timeseries.tsv'.format(bd=data_dir, c=c, s=stim)
            if stim == 'deactivating':
                new_data = get_data(ts_file, c, n_timeseries, reps, -p_labels, t=times)
                # Don't add the 0 perturbation for deactiving, as it is redundant
                new_data.drop(0, level='perturb', inplace=True)
            else:
                new_data = get_data(ts_file, c, n_timeseries, reps, p_labels, t=times)
            data = pd.concat([data, new_data])
    data.sort_index(inplace=True)
    data.index.rename('time', 3, inplace=True)

    df = pd.read_csv(path, sep='\t')
    df['condition'] = c
    times = sorted(list(set(df['Time'].values)))

    # For safety
    if not n_timeseries.is_integer():
        raise ValueError('Number of time points for each replicate is not the same')

    p_rep_list = np.array(list(range(reps)) * int(n_timeseries))
    ts_p_index = np.ceil((df.index.values + 1) / len(times)).astype(int) - 1
    ts_rep_list = p_rep_list[ts_p_index]
    ts_p_list = perturbation_labels[ts_p_index]

    df['perturb'] = ts_p_list
    df['replicate'] = ts_rep_list

    idx = pd.IndexSlice
    full_data = df.set_index(['condition', 'replicate', 'perturb', 'Time']).sort_index()

    if t is None:
        t = full_data.index.levels[full_data.index.names.index('Time')].values

    return full_data.loc[idx[:, :, :, t], :].copy()


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
        shuffled = grouped_data.apply(shuffle)  # type: pd.DataFrame
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
        _, cur_scores = fit_dea(p_data, contrast, reference_labels=['condition', 'time'], log2=True)

        # drop cluster column, rename score, and add to real scores
        cur_scores.drop('Cluster', inplace=True, axis=1)
        cur_scores.columns = ['p{}_score'.format(p_idx)]
        p_score = pd.concat([p_score, cur_scores], axis=1)

    p_mean = p_score.mean(axis=1)
    p_std = p_score.std(axis=1)
    values = real_scores['score'].copy()
    p_z = -1 * ((values - p_mean) / p_std).abs()
    p_values = (2 * p_z.apply(stats.norm.cdf)).round(decimals=int(n_permutes / 10))
    p_values.name = 'p_value'
    new_scores = pd.concat([real_scores, p_values], axis=1)  # type: pd.DataFrame

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


def compile_sim(sim_dir, times, save_path=None, pp=True, **kwargs):
    # Initialize the results object
    gnr = GnwNetResults(sim_dir, **kwargs)

    print("Compiling simulation results. This could take a while")
    sim_results = gnr.compile_results(censor_times=times, save_intermediates=False, pp=pp)
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
    scores.set_index(['perturbation', 'id'], append=True, inplace=True)
    scores = scores.swaplevel(i='id', j='gene').sort_index()

    return scores


def correlate(experimental: pd.DataFrame, simulated: pd.DataFrame, sim_conditions, sim_node=None):
    # Get group means and zscore
    gene_mean = experimental.groupby(level=['condition', 'time'], axis=1).mean()
    mean_z = gene_mean.groupby(level='condition', axis=1).transform(stats.zscore, ddof=1).fillna(0)

    # Correlate zscored means for each gene with each node in every simulation
    if sim_node is not None:
        simulated = simulated[simulated.index.get_level_values('gene') == sim_node]
    sim_means = simulated.loc[:, sim_conditions]
    sim_mean_z = sim_means.groupby(level='stat', axis=1).transform(stats.zscore, ddof=1).fillna(0)
    sim_mean_z.columns.set_levels([c.replace('_mean', "") for c in sim_mean_z.columns.levels[0]], level=0, inplace=True)
    corr = []
    for c in ['ko', 'wt']:
        print('Computing pairwise for {}'.format(c))
        pcorr, p = pairwise_corr(sim_mean_z.loc[:, c], mean_z.loc[:, c], axis=1)
        corr.append(pcorr)
    print('Done')
    return (corr[0] + corr[1]) / 2


def display_sim(network, stim, perturbation, times, directory, exp_condition='ko', ctrl_condition='wt'):
    data_dir = '{}/{}/{}/'.format(directory, network, stim)
    network_structure = "{}/{}/{}_goldstandard_signed.tsv".format(directory, network, network)

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


def match_to_gene(x, y, correlation):
    matching_results = pd.DataFrame()
    for gene, row in x.iterrows():
        candidate_nets = y.loc[y.Cluster == row.Cluster]
        cur_corr = correlation.loc[gene, candidate_nets.index]
        cur_corr.name = 'pearson_r'
        ranking = pd.concat([candidate_nets, cur_corr], axis=1)
        ranking['mean'] = (ranking['score'] + ranking['pearson_r']) / 2
        ranking = ranking.loc[ranking.index.get_level_values(2) == 'y']
        ranking['true_gene'] = gene
        matching_results = pd.concat([matching_results, ranking.reset_index()], ignore_index=True, join='inner')

    return matching_results


def clustering_hamming(x, y):
    ham = [hamming(cluster, y[ii]) for ii, cluster in enumerate(x)]
    return ham


def match_true_predicted_data(matching, true_data, predicted_data):
    print(matching)


def compile_match_sim_data(matching, base_dir, condition='ki', times=None):
    compile_args = [(ii, base_dir, condition, times) for ii in matching.index.unique()]
    pool = mp.Pool()
    info = pool.starmap(get_sim_data, compile_args)
    pool.close()
    pool.join()
    sim_data = pd.concat(info, axis=1)      # type: pd.DataFrame
    return sim_data


def get_sim_data(sim_tuple, directory, condition='ki', times=None):
    net = sim_tuple[0]
    print(net)
    perturb = abs(sim_tuple[1])
    mode = 'activating' if sim_tuple[1] >= 0 else 'deactivating'
    node = sim_tuple[2]
    data_dir = "{}/{}/{}/".format(directory, net, mode)
    gsr = GnwSimResults(data_dir, net, condition, sim_suffix='dream4_timeseries.tsv',
                        perturb_suffix="dream4_timeseries_perturbations.tsv")
    idx = pd.IndexSlice
    if times is not None:
        series = gsr.annotated_data.loc[idx[:, :, perturb, times], node]
    else:
        series = gsr.annotated_data.loc[idx[:, :, perturb, :], node]

    # Add a name to make concationation easier
    series.name = str(sim_tuple)

    # Drop the perturbation so NaNs aren't made in the final DataFrame
    series.index = series.index.droplevel('perturbation')
    return series

if __name__ == '__main__':
    pd.set_option('display.width', 250)
    override = False    # Rerun certain parts of the analysis
    ### TRAINING

    ## Load the data
    # Set project parameters
    t = [0, 15, 30, 60, 120, 240, 480]
    conditions = ['ko', 'wt', 'ki']
    stimuli = ['activating', 'deactivating']
    reps = 3
    project_name = 'insilico_strongly_connected_2'
    data_dir = '../data/insilico/strongly_connected_2/'


    # Keep track of the gene names
    gene_names = pd.read_csv("{}gene_anonymization.csv".format(data_dir), header=None, index_col=0)
    ko_gene = 'YMR016C'
    stim_gene = 'YKL062W'
    ko_gene = gene_names.loc[ko_gene, 1]
    stim_gene = gene_names.loc[stim_gene, 1]

    # Keep track of the perturbations
    perturbations = pd.read_csv("{}perturbations.csv".format(data_dir), index_col=0)
    p_labels = perturbations.index.values
    n_timeseries = len(p_labels) / reps
    df, dg = tsv_to_dg("{}Yeast-100_anon_goldstandard_signed.tsv".format(data_dir))

    data = load_insilico_data(data_dir, conditions, stimuli, 'Yeast-100_anon', times=t)
    sys.exit()

    idx = pd.IndexSlice
    perturb = 1
    raw = data.loc[idx[:, :, perturb, :], :].T
    contrast = 'ko-wt'
    prefix = prefix = "{}/{}_{}_".format(project_name, project_name, contrast)

    # Mean-variance plot
    # plt.plot(raw.mean(axis=1), raw.std(axis=1), '.')
    # plt.show()

    try:
        dea = pd.read_pickle("{}dea.pkl".format(prefix))  # type: DEAnalysis
    except:
        dea = dde(raw, contrast, project_name, save_permute_data=True, calc_p=True, voom=False, log2=True)

    if override:
        dea = dde(raw, contrast, project_name, save_permute_data=True, calc_p=True, voom=False, log2=True)

    filtered_data = dea.data.loc[:, contrast.split('-')]
    der = dea.results[contrast]

    scores = pd.read_pickle("{}dde.pkl".format(prefix))

    dde_genes = filter_dde(scores, thresh=2).sort_values('score', ascending=False)
    dde_genes.sort_values('score', ascending=False, inplace=True)

    try:
        sim_stats = pd.read_pickle("{}sim_stats.pkl".format(prefix))  # type: pd.DataFrame
    except:
        sim_stats = compile_sim('../data/motif_library/gnw_networks/', times=t,
                                save_path="{}sim_stats.pkl".format(prefix))

    idx = pd.IndexSlice
    sim_stats.sort_index(inplace=True)
    sim_scores = discretize_sim(sim_stats.loc[idx[:, :, 'y'], :].copy(), filter_interesting=False)

    try:
        corr = pd.read_pickle("{}data_to_sim_corr.pkl".format(prefix))
    except:
        corr = correlate(filtered_data, sim_stats.loc[sim_scores.index], ['ko_mean', 'wt_mean'], sim_node='y')
        corr.to_pickle("{}data_to_sim_corr.pkl".format(prefix))

    match = match_to_gene(dde_genes, sim_scores, corr)
    match.set_index(['id', 'perturbation', 'gene'], inplace=True)
    match.sort_values('mean', ascending=False, inplace=True)

    ### TESTING
    true_prefix = prefix.replace('ko', 'ki')
    contrast = contrast.replace('ko', 'ki')
    try:
        true_dea = pd.read_pickle("{}dea.pkl".format(true_prefix))  # type: DEAnalysis
    except:
        true_dea = dde(raw, contrast, project_name, save_permute_data=True, calc_p=True, voom=False, log2=True)
    if override:
        true_dea = dde(raw, contrast, project_name, save_permute_data=True, calc_p=True, voom=False, log2=True)

    der = true_dea.results[contrast]
    true_scores = pd.read_pickle("{}dde.pkl".format(true_prefix))
    true_dde_genes = filter_dde(true_scores)
    true_dde_genes = true_dde_genes.sort_values('score', ascending=False)

    true_filtered_data = true_dea.data.loc[true_dde_genes.index, contrast.split('-')]

    test_sim = pd.read_pickle("{}sim_stats.pkl".format(true_prefix))  # type: pd.DataFrame
    test_scores = discretize_sim(test_sim, filter_interesting=False)
    test_scores = test_scores.loc[match.index]
    test_scores.columns = ['predicted_cluster', 'predicted_score']

    true_scores = pd.read_pickle("{}dde.pkl".format(true_prefix))
    true_dea = pd.read_pickle("{}dea.pkl".format(true_prefix))  # type: DEAnalysis
    true_der = true_dea.results[contrast]  # type: DEResults

    matched_genes = list(set(match.true_gene))
    unique_nets = list(set(match.index))
    true_dde_genes = filter_dde(true_scores)
    true_dde_genes = true_dde_genes[true_dde_genes['p_value'] < 0.05].sort_values('score', ascending=False)
    true_filtered_data = true_dea.data.loc[true_dde_genes.index, contrast.split('-')]

    combined = pd.concat([match, test_scores], join='inner', axis=1)

    # Some networks aren't affected by knockin
    combined.dropna(inplace=True)
    combined.sort_index(inplace=True)

    try:
        matched_sim_data = pd.read_pickle('{}_matched_ki_sim_data.pkl'.format(true_prefix)).T
    except:
        matched_sim_data = compile_match_sim_data(combined, "../data/motif_library/gnw_networks/", times=t)
        matched_sim_data.to_pickle('{}_matched_ki_sim_data.pkl'.format(true_prefix))
    if override:
        matched_sim_data = compile_match_sim_data(combined, "../data/motif_library/gnw_networks/", times=t)
        matched_sim_data.to_pickle('{}_matched_ki_sim_data.pkl'.format(true_prefix))

    combined.sort_values('mean', ascending=False, inplace=True)
    for i in range(5):
        pred = matched_sim_data.loc[str(combined.index.values[i])]
        pz = stats.zscore(pred, ddof=1)
        test = data.T.loc[combined.true_gene[i], idx['ki', :, combined.index.get_level_values('perturbation')[i], :]]
        tz = stats.zscore(test, ddof=1)
        test1 = data.T.loc[combined.true_gene[i], idx['ki', :, 1, :]]
        plt.figure()
        plt.plot(pred.index.get_level_values('Time'), np.log2(pred.values), '.', label='pred')
        plt.plot(test.index.get_level_values('time'), np.log2(test.values), '.', label='test')
        plt.plot(test1.index.get_level_values('time'), np.log2(test1.values), '.', label='test1')
        plt.legend()

        # plt.figure()
        # plt.plot(pred.index.get_level_values('Time'), pz, '.')
        # plt.plot(test.index.get_level_values('time'), tz, '.')
    plt.show()
    sys.exit()

    print(combined.sort_values('mean', ascending=False).head())
    m = matched_sim_data.loc[[str(ii) for ii in combined.index.values]] # type: pd.DataFrame
    m.columns.set_levels(['sim_value'], level='condition', inplace=True)
    m.columns.set_names(['condition', 'replicate', 'time'], inplace=True)
    true_data = true_dea.data.loc[:, idx['ki', :, 1]]
    true_data.columns.set_levels(['true_value'], level='condition', inplace=True)
    true_data.columns.set_levels([1,2,3], level='replicate', inplace=True)
    tt = true_data.loc[combined['true_gene']]
    tt.columns = tt.columns.droplevel('perturb')
    new_index = list(zip(m.index, tt.index))
    tt.index = new_index
    m.index = new_index
    # m = m.apply(stats.zscore, axis=1, ddof=1)
    # tt = tt.apply(stats.zscore, axis=1, ddof=1)
    test_data = pd.concat([m, tt], axis=1)
    print(sum(test_data.index.duplicated()))
    test_dea = DEAnalysis(test_data, reference_labels=['condition', 'time'], log2=True)
    test_dea.fit_contrasts(test_dea.default_contrasts['sim_value-true_value']['contrasts'], fit_names='test')
    test_der = test_dea.results['test']
    print(test_der.top_table(p=0.05).shape)
    print(test_der.top_table().shape)
    print(test_der.top_table().head())
    # print(test_der.top_table().loc["('350', 0.75, 'y')", 'G57'])
    # display_sim('350', 'activating', .75, t, '../data/motif_library/gnw_networks/')
    # dep = DEPlot()
    # dep.tsplot(true_dea.data.loc['G57'])
    # plt.show()




