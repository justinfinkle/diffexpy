import ast
import itertools as it
import multiprocessing as mp
import os
import shutil
import sys
import tarfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pydiffexp import DEAnalysis, DEResults, get_scores, cluster_discrete, pairwise_corr
from pydiffexp.gnw import mk_ch_dir, GnwNetResults, GnwSimResults, draw_results, get_graph
from scipy import stats
from scipy.spatial.distance import hamming
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


def load_insilico_data(path, conditions, stimuli, net_name, times=None) -> pd.DataFrame:
    """

    :param path:
    :param conditions:
    :param stimuli:
    :param net_name:
    :param times:
    :return:
    """
    # List comprehension: for each combo of stimuli and conditions make a GSR object and get the data
    df_list = []
    for ss, cc in it.product(stimuli, conditions):
        c_df = GnwSimResults('{}/{}/'.format(path, ss), net_name, cc, sim_suffix='dream4_timeseries.tsv',
                                   perturb_suffix="dream4_timeseries_perturbations.tsv", censor_times=times).data
        if ss == 'deactivating':
            c_df.index = c_df.index.set_levels(-c_df.index.levels[c_df.index.names.index('perturbation')],
                                               'perturbation')
        df_list.append(c_df)

    # Concatenate into one dataframe
    insilico_data = pd.concat(df_list)

    return insilico_data.sort_index()


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


def analyze_permutes(real_scores, permutes_path, contrast, **kwargs) -> (pd.DataFrame, pd.DataFrame):
    p_score = pd.DataFrame()
    n_permutes = len(os.listdir(permutes_path))
    for p in os.listdir(permutes_path):
        print(p)
        # Load data and fit to get permuted data p-values
        p_idx = p.split("_")[0]
        p_data = pd.read_pickle(os.path.join(permutes_path, p))
        _, cur_scores = fit_dea(p_data, contrast, **kwargs)

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
        data = dea.raw_data.loc[:, idx[default_contrast.split('-'), :, :]]
        grouped = data.groupby(level='condition', axis=1)
        save_permutes(permute_path, grouped, n=n_permutes)

    if calc_p and len(os.listdir(permute_path)):
        print("Calculating cluster ranking pvalues")
        scores, ptest_scores = analyze_permutes(scores, permute_path, default_contrast, **kwargs)
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


def filter_dde(df, col='Cluster', thresh=2, p=0.05, s=0):
    """
    Filter out dde genes that are "uninteresting". They have fewer unique discrete labels than the threshold
    e.g. thresh=2 and Cluster = (1,1,1,1) will have only 1 unique label, and will be filtered out.
    :param df:
    :param col:
    :param thresh:
    :return:
    """
    df = df.loc[df[col].apply(ast.literal_eval).apply(set).apply(len) >= thresh]
    df = df[(df.score > s) & (df.p_value < p)]
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
    for c in ['ki', 'wt']:
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
    draw_results(np.log2(data+1), perturbation, titles, times=times, g=dg)
    plt.tight_layout()


def match_to_gene(x, y, correlation, unique_net=True):
    matching_results = pd.DataFrame()
    for gene, row in x.iterrows():
        candidate_nets = y.loc[y.Cluster == row.Cluster]
        cur_corr = correlation.loc[gene, candidate_nets.index]
        cur_corr.name = 'pearson_r'
        ranking = pd.concat([candidate_nets, cur_corr], axis=1)
        ranking['mean'] = (ranking['score'] + ranking['pearson_r']) / 2
        ranking = ranking.loc[ranking.index.get_level_values(2) == 'y'].copy()

        # Remove same network ids that are just different perturbations
        if unique_net:
            sorted_ranking = ranking.sort_values('mean', ascending=False)
            ranking = sorted_ranking[~sorted_ranking.index.get_level_values('id').duplicated(keep='first')].copy()
        ranking['true_gene'] = gene
        matching_results = pd.concat([matching_results, ranking.reset_index()], ignore_index=True)

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
    # Options
    pd.set_option('display.width', 250)
    override = False  # Rerun certain parts of the analysis
    plot_mean_variance = False

    """
    ===================================
    ========== Load the data ==========
    ===================================
    """
    # Prep the raw data
    project_name = "GSE69822"
    contrast = 'ki-wt'
    t = [0, 15, 40, 90, 180, 300]
    raw = load_data('../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt')
    gene_map = pd.read_csv('../data/GSE69822/mcf10a_gene_names.csv', index_col=0)
    mk_ch_dir(project_name, ch=False)

    """
    ===================================
    ============ TRAINING =============
    ===================================
    """
    contrast = 'ki-wt'
    prefix = "{}/{}_{}_".format(project_name, project_name, contrast) # For saving intermediate data

    try:
        dea = pd.read_pickle("{}dea.pkl".format(prefix))  # type: DEAnalysis
    except:
        dea = dde(raw, contrast, project_name, save_permute_data=True, calc_p=True, voom=True)

    if override:
        dea = dde(raw, contrast, project_name, save_permute_data=True, calc_p=True, voom=True)

    scores = pd.read_pickle("{}dde.pkl".format(prefix))

    # Mean-variance plot
    if plot_mean_variance:
        plt.plot(dea.data.mean(axis=1), dea.data.std(axis=1), '.')
        plt.xlabel('Mean expression')
        plt.ylabel('Expression std')
        plt.title('Heteroskedasticity')
        plt.tight_layout()
        plt.show()

    filtered_data = dea.data.loc[:, contrast.split('-')]
    der = dea.results[contrast]

    dde_genes = filter_dde(scores, thresh=2, p=1).sort_values('score', ascending=False)
    dde_genes.sort_values('score', ascending=False, inplace=True)

    try:
        sim_stats = pd.read_pickle("{}sim_stats.pkl".format(prefix))  # type: pd.DataFrame
    except:
        sim_stats = compile_sim('../data/motif_library/gnw_networks/', times=t,
                                save_path="{}sim_stats.pkl".format(prefix))

    idx = pd.IndexSlice
    sim_stats.sort_index(inplace=True)
    sim_stats = sim_stats.loc[idx[:, 1, :], :].copy()
    sim_scores = discretize_sim(sim_stats, filter_interesting=False)

    try:
        corr = pd.read_pickle("{}data_to_sim_corr.pkl".format(prefix))
    except:
        corr = correlate(filtered_data, sim_stats.loc[sim_scores.index], ['ki_mean', 'wt_mean'], sim_node='y')
        corr.to_pickle("{}data_to_sim_corr.pkl".format(prefix))

    match = match_to_gene(dde_genes, sim_scores, corr, unique_net=False)
    match.set_index(['id', 'perturbation', 'gene'], inplace=True)
    match.sort_values('mean', ascending=False, inplace=True)
    match = match[match['mean'] > 0]
    # match = match[~match.true_gene.duplicated(keep='first')]
    matched_genes = list(set(match.true_gene))
    unique_nets = list(set(match.index))
    print(len(matched_genes))

    """
    ====================================
    ============= TESTING ==============
    ====================================
    """
    test_prefix = prefix.replace('ko', 'ki')
    test_contrast = contrast.replace('ko', 'ki')

    # Predict how the gene will respond compared to the WT
    try:
        true_dea = pd.read_pickle("{}dea.pkl".format(test_prefix))  # type: DEAnalysis
    except:
        true_dea = dde(raw, test_contrast, project_name, save_permute_data=True, calc_p=True, voom=True)

    predicted_scores = pd.read_pickle("{}dde.pkl".format(test_prefix))

    try:
        pred_sim = pd.read_pickle("{}sim_stats.pkl".format(test_prefix))  # type: pd.DataFrame
    except:
        pred_sim = compile_sim('../data/motif_library/gnw_networks/', times=t,
                               save_path="{}sim_stats.pkl".format(test_prefix), experimental='ki')
    idx = pd.IndexSlice
    pred_sim.sort_index(inplace=True)
    pred_sim = pred_sim.loc[idx[:, 1, 'y'], :]
    true_data = dea.data.loc[:, idx['ki', :, :]].groupby(level='time', axis=1).mean()
    match['corr'] = match.apply(lambda x: stats.pearsonr(true_data.loc[x.true_gene], pred_sim.loc[x.name, 'ki_mean'])[0], axis=1)

    pred_wlfc = (pred_sim.loc[:, 'lfc'])#*(1-pred_sim.loc[:, 'lfc_pvalue']))
    true_wlfc = (true_dea.results['ki-wt'].top_table().iloc[:, :len(t)])# * (1-true_dea.results['ki-wt'].p_value))

    match['mae'] = match.apply(lambda x: mse(true_wlfc.loc[x.true_gene], pred_wlfc.loc[x.name]), axis=1)
    # sns.violinplot(data=match, x='true_gene', y='mae')
    # plt.show()
    # sys.exit()
    pred_clusters = discretize_sim(pred_sim, filter_interesting=False)
    reduced_set = True
    if reduced_set:
        # pred_wlfc = pred_wlfc.loc[~match.index.duplicated()]
        true_wlfc = true_wlfc.loc[matched_genes]

    gene_mae_dist_dict = {ii: [mse(pwlfc, twlfc) for pwlfc in pred_wlfc.values] for ii, twlfc in true_wlfc.iterrows()}

    # mae_dist, pear_dist = zip(*[(mae(twlfc, pwlfc), stats.pearsonr(twlfc, pwlfc)) for twlfc in true_wlfc.values for pwlfc in pred_wlfc.values])

    # plt.hist([mae(twlfc, pwlfc) for twlfc in true_wlfc.values for pwlfc in pred_wlfc.values], log=True)
    # plt.show()
    # sys.exit()
    resamples = 100
    g = match.groupby('true_gene')
    # print(g['mean'].mean())
    # plt.plot(g['mean'].mean(), g['mae'].mean(), '.')
    # plt.show()
    # sys.exit()

    sig = 0
    n_matches = len(g.groups)
    diffs = []


    def sample_stats(df, dist_dict, resamples=100):
        random_sample_means = [np.mean(np.random.choice(dist_dict[df.name], len(df))) for _ in range(resamples)]
        rs_mean = np.median(random_sample_means)
        mean_lfc_mae = mae(true_wlfc.loc[df.name], pred_wlfc.loc[g.get_group(df.name).index].mean(axis=0))
        ttest = stats.mannwhitneyu(df.mae, random_sample_means)
        s = pd.Series([len(df), (rs_mean - df.mae.median()) / rs_mean * 100, ttest.pvalue / 2, mean_lfc_mae, rs_mean,
                       df.mae.median()],
                      index=['n', 'mae_diff', 'mae_pvalue', 'mean_lfc_mae', 'random_mae', 'group_mae'])
        return s


    x = pd.concat([dde_genes, g.apply(sample_stats, gene_mae_dist_dict, resamples)], axis=1)
    sig = x[(x.mae_diff > 0) & (x.mae_pvalue < 0.05)].sort_values('mae_pvalue')
    print(sig)
    sns.boxplot(data=pd.melt(sig, id_vars=sig.columns[:-3], value_vars=sig.columns[-3:]), x='variable', y='value',
                notch=True, showfliers=False)
    plt.tight_layout()
    # plt.show()
    print(sig.mae_diff.mean())
    print(stats.mannwhitneyu(sig.mean_lfc_mae, sig.random_mae))
    print(stats.mannwhitneyu(sig.group_mae, sig.random_mae))
    sys.exit()
            # if ttest.pvalue/2< 0.05 and info.mae.mean()<rs_mean: # Significant
            #     sig +=1

    print(sig)
    sys.exit()
    print(stats.fisher_exact(ct))
    sns.regplot(g['mean'].mean().values, np.array(diffs))
    plt.show()
    sns.swarmplot(diffs)
    plt.show()
        # print(ii, len(info), info.mae.mean(), np.mean(mae_dist), stats.mannwhitneyu(info.mae, mae_dist).pvalue<0.05)
    sys.exit()
    # clusters = np.array(list(map(eval, set(pred_clusters.loc[idx[:, :, 'y'], 'Cluster'])))).astype(int)
    # # clusters = np.array(list(map(eval, pred_clusters.loc[idx[:, :, 'y'], 'Cluster']))).astype(int)
    # p_clusters = np.array(list(map(eval, set(predicted_scores.Cluster)))).astype(int)
    # for pc in p_clusters:
    #     hams = [hamming(cc, pc) for cc in clusters]
    #     print(len(hams), set(hams))
    # sys.exit()
    # ham_dist = {}
    # for cc in np.array(list(map(eval, set(predicted_scores.Cluster)))).astype(int):
    #     hams = [hamming(cc, pc) for pc in clusters]
    #     print(len(set(hams)))
    #     ham_dist[tuple(cc)] = [hamming(cc, pc) for pc in clusters]
    # sys.exit()
    # match['predicted_cluster'] = list(map(lambda x: np.array(eval(x)), pred_clusters.loc[match.index, 'Cluster']))
    # match['actual_cluster'] = list(map(lambda x: np.array(eval(x)), predicted_scores.loc[match.true_gene, 'Cluster'].values))
    # match['ham'] = [hamming(info.predicted_cluster, info.actual_cluster) for ii, info in match.iterrows()]
    #
    # for ii, info in g:
    #     true_dist = ham_dist[tuple(info.actual_cluster[0])]
    #     print(ii, stats.mannwhitneyu(info.ham, true_dist, alternative='greater'))
    sys.exit()

    try:
        matched_sim_data = pd.read_pickle('{}matched_ki_sim_data.pkl'.format(test_prefix)).T
    except:
        matched_sim_data = compile_match_sim_data(match, "../data/motif_library/gnw_networks/", times=t)
        matched_sim_data.to_pickle('{}matched_ki_sim_data.pkl'.format(test_prefix))
    if override:
        matched_sim_data = compile_match_sim_data(match, "../data/motif_library/gnw_networks/", times=t)
        matched_sim_data.to_pickle('{}matched_ki_sim_data.pkl'.format(test_prefix))

    matched_sim_data.index = pd.MultiIndex.from_tuples(list(map(eval, matched_sim_data.index)),
                                                       names=['id', 'perturbation', 'gene'])

    predictions = matched_sim_data.loc[match.index]                                   # type: pd.DataFrame
    predictions.columns.set_levels(['sim_value'], level='condition', inplace=True)
    predictions.columns.set_names(['condition', 'replicate', 'time'], inplace=True)

    predictions.index = match.set_index('true_gene', append=True).index

    true_data = raw.loc[:, idx['ki', :, 1]]-1
    true_data.columns.set_levels(['true_value'], level='condition', inplace=True)
    true_data.columns.set_levels([1, 2, 3], level='replicate', inplace=True)
    tt = true_data.loc[match['true_gene']]
    tt.columns = tt.columns.droplevel('perturb')
    tt.index = predictions.index

    tt = tt.divide(tt.abs().max(axis=1), axis=0)
    predictions = predictions.divide(predictions.abs().max(axis=1), axis=0)

    test_data = pd.concat([predictions + 1, tt+1], axis=1)
    test_dea = DEAnalysis(test_data, reference_labels=['condition', 'time'], log2=True)

    test_dea.fit_contrasts(test_dea.default_contrasts['sim_value-true_value']['contrasts'], fit_names='test')
    test_der = test_dea.results['test']

    top = test_der.top_table()
    # print(test_der.top_table().shape)
    # print(test_der.top_table(p=0.05).shape)
    # print(test_der.top_table())
    # sys.exit()
    top.index = pd.MultiIndex.from_tuples(list(map(eval, top.index)), names=predictions.index.names)
    # for ii in top.index.values:
    #     dep.tsplot(np.log2(test_data.loc[ii]))
    #     plt.tight_layout()
    #     plt.show()
    # print(top.index.values[0])
    # sys.exit()

    mean = match['mean']
    p = top.loc[predictions.index, 'adj_pval']
    plt.plot(mean, p, '.')
    plt.show()
    sys.exit()
    # print(test_der.top_table().loc[str((str(('1335', .75, 'y')), 'G60'))])
    # print(test_der.p_value)
    # sys.exit()
    # print(match.head())

    pred = (get_sim_data((1335, .75, 'y'), '../data/motif_library/gnw_networks/', 'ki', t)+1)
    true = true_data.loc['G60']
    dep.tsplot(np.log2(pred), subgroup='Time')
    dep.tsplot(np.log2(true))
    plt.show()
    sys.exit()
    display_sim('1845', 'activating', 1, t, '../data/motif_library/gnw_networks/', exp_condition='ki')
    dep.tsplot(np.log2(raw.loc['G92']))
    plt.tight_layout()
    plt.show()




