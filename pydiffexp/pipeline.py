import sys
import ast
import multiprocessing as mp
import os
import shutil
import tarfile

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pydiffexp import DEAnalysis, DEResults, DEPlot, get_scores, cluster_discrete, pairwise_corr
from pydiffexp.gnw import mk_ch_dir, GnwNetResults, GnwSimResults, draw_results, get_graph
from scipy import stats
from scipy.spatial.distance import hamming


class DynamicDifferentialExpression(object):
    """
    Coordinate training and testing of DDE models
    """
    def __init__(self, directory, permute=True):
        self.dir = None
        self.permute = permute
        self.training = {}          # type: dict
        self.test = []
        self.estimators = None      # type: pd.core.groupby.DataFrameGroupBy
        self.dea = None             # type: DEAnalysis
        self.sim_stats = None       # type: pd.DataFrame
        self.corr = None            # type: pd.DataFrame
        self.sim_scores = None      # type: pd.DataFrame
        self.match = None           # type: pd.DataFrame
        self.dde_genes = None       # type: pd.DataFrame
        self.times = None           # type: list

        # There is a lot of intermediary data that can be saved to make rerunning the analysis easier
        self.set_save_directory(directory)

    def set_save_directory(self, path):
        """
        Set the default path for saving intermediate files
        :param path: path-like
        :return:
        """
        os.path.isdir(path)
        self.dir = os.path.abspath(path)
        return

    def set_training_conditions(self, exp, ctrl):
        self.training = {'experimental': exp, 'control': ctrl}
        return

    def set_test_conditions(self, t):
        self.test.append(t)
        return

    @staticmethod
    def load_sim_stats(path, times, experimental, control, **kwargs):
        try:
            sim_stats = pd.read_pickle(path)  # type: pd.DataFrame

        except:
            sim_stats = compile_sim('../data/motif_library/gnw_networks/',
                                    times=times, save_path=path,
                                    experimental=experimental, control=control)

        # Reduce the dataframe if a slicer is passed
        try:

            # Filter the axes
            mi_filter = [kwargs[level] if level in kwargs.keys() else slice(None)
                         for level in sim_stats.index.names]
            sim_stats = sim_stats.loc[tuple(mi_filter), :].copy()

        except TypeError:
            pass

        return sim_stats

    @staticmethod
    def _sim_filter_default():
        """
        Default args for the simulation results
        :return:
        """
        return {'perturbation': 1, 'gene': 'y'}

    def predict(self, test, prefix, ctrl=None, sim_filter=None):
        if sim_filter is None:
            sim_filter = self._sim_filter_default()

        self.set_test_conditions(test)
        if ctrl is None:
            ctrl = self.training['control']

        contrast = '{}-{}'.format(test, ctrl)

        sim_path = os.path.join(self.dir,
                                '{}_{}_sim_stats.pkl'.format(prefix, contrast))

        # Get the predicted values
        pred_stats = self.load_sim_stats(sim_path, self.times, test, ctrl,
                                         **sim_filter)

        # Calculate estimator predictions
        prediction = self.estimators.apply(self.estimator_prediction, ctrl=ctrl,
                                           pred_lib=pred_stats)

        return prediction

    def estimator_prediction(self, df, ctrl, pred_lib):
        """
        Calculate the prediction for a new condition with a set of trained
        estimators. Meant to work with groubpy.apply()
        :param df:
        :param ctrl: the control condition
        :param pred_lib:
        :return:
        """
        # Calculate the baseline
        baseline = self.dea.data.loc[df.name, ctrl].groupby('time').mean()

        # Calculate the average log fold change prediction in the group
        pred_lfc = pred_lib.loc[df.index, 'lfc'].mean()

        pred = baseline+pred_lfc
        return pred

    def compare_random(self, ):
        pass

    def random_sample(self, df, dist_dict, resamples=100):
        random_sample_means = [np.mean(np.random.choice(dist_dict[df.name], len(df))) for _ in range(resamples)]
        rs_mean = np.median(random_sample_means)
        mean_lfc_mae = mse(true_wlfc.loc[df.name], pred_wlfc.loc[g.get_group(df.name).index].mean(axis=0))
        ttest = stats.mannwhitneyu(df.mae, random_sample_means)
        s = pd.Series([len(df), (rs_mean - df.mae.median()) / rs_mean * 100, ttest.pvalue / 2, mean_lfc_mae, rs_mean,
                       df.mae.median()],
                      index=['n', 'mae_diff', 'mae_pvalue', 'mean_lfc_mae', 'random_mae', 'group_mae'])
        return s

    def train(self, data, prefix, times=None, override=False, experimental='ko',
              control='wt', sim_filter=None, **kwargs):
        """
        Train DDE estimators
        :param data:
        :param prefix:
        :param times:
        :param override:
        :param experimental:
        :param control:
        :param sim_filter: dict; filters out simulations from the multiindex.
        Default is None, which keeps only simulations for perturbation=1
        and gene='y'
        :param kwargs:
        :return:
        """

        if sim_filter is None:
            sim_filter = self._sim_filter_default()

        # Set conditions
        contrast = "{}-{}".format(experimental, control)
        self.set_training_conditions(experimental, control)

        # Define paths to save or read pickles from
        dea_path = os.path.join(self.dir, '{}_{}_dea.pkl'.format(prefix, contrast))
        scores_path = os.path.join(self.dir, '{}_{}_scores.pkl'.format(prefix, contrast))
        sim_path = os.path.join(self.dir, '{}_{}_sim_stats.pkl'.format(prefix, contrast))
        corr_path = os.path.join(self.dir, '{}_{}_data_to_sim_corr.pkl'.format(prefix, contrast))

        # Try to load the DEAnalysis pickle
        try:
            # If the user wants to override, raise a ValueError to force exception
            if override:
                raise ValueError('Override to retrain')

            # Load the dea pickle
            dea = pd.read_pickle(dea_path)  # type: DEAnalysis

            # Load the scores pickle
            scores = pd.read_pickle(scores_path)

        # Rerun the analysis
        # todo: cleanup this function
        except (FileNotFoundError, ImportError, ValueError) as e:
            dea, scores = self.dde(data, contrast, self.dir, **kwargs)

        dde_genes = filter_dde(scores, thresh=2, p=1).sort_values('Cluster', ascending=False)
        filtered_data = dea.data.loc[:, contrast.split('-')]

        if times is None:
            times = dea.times

        sim_stats = self.load_sim_stats(sim_path, times,
                                        experimental=experimental,
                                        control=control,
                                        **sim_filter)

        sim_stats.sort_index(inplace=True)

        sim_scores = discretize_sim(sim_stats, filter_interesting=False)

        try:
            corr = pd.read_pickle(corr_path)
        except:
            corr = self.correlate(filtered_data, sim_stats.loc[sim_scores.index], sim_node='y')
            corr.to_pickle(corr_path)

        match = match_to_gene(dde_genes, sim_scores, corr, unique_net=False)
        match.set_index(['id', 'perturbation', 'gene'], inplace=True)
        match.sort_values('mean', ascending=False, inplace=True)
        match = match[match['mean'] > 0]

        self.times = times
        self.estimators = match.groupby('true_gene')
        self.match = match
        self.sim_stats = sim_stats
        self.corr = corr
        self.sim_scores = sim_scores
        self.dea = dea
        self.dde_genes = dde_genes

        return match

    # todo: clean this up and make it a class method
    def dde(self, data, default_contrast, project_dir, n_permutes=100,
            permute_path=None, save_permute_data=False, calc_p=False,
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
        scores.to_pickle("{}scores.pkl".format(prefix))

        if save_permute_data:
            print("Creating permute data")
            # Make a directory for the permutes
            if permute_path is None:
                permute_path = "{}permutes".format(prefix)
            mk_ch_dir(permute_path, ch=False)

            idx = pd.IndexSlice
            data = dea.raw_data.loc[:, idx[default_contrast.split('-'), :, :]]
            grouped = data.groupby(level='condition', axis=1)
            save_permutes(permute_path, grouped, n=n_permutes)

            if compress:
                print('Compressing permute directory')
                compress_directory(permute_path)
                print('Removing permutes directory')
                shutil.rmtree(permute_path)

        if calc_p and len(os.listdir(permute_path)):
            print("Calculating cluster ranking pvalues")
            scores, ptest_scores = analyze_permutes(scores, permute_path, default_contrast, **kwargs)
            print('Saving permutation test results')
            scores.to_pickle("{}dde.pkl".format(prefix))
            ptest_scores.to_pickle("{}ptest_scores.pkl".format(prefix))

        return dea, scores

    def correlate(self, experimental: pd.DataFrame, simulated: pd.DataFrame, sim_node=None):
        # Get group means and zscore
        gene_mean = experimental.groupby(level=['condition', 'time'], axis=1).mean()
        mean_z = gene_mean.groupby(level='condition', axis=1).transform(stats.zscore, ddof=1).fillna(0)

        # Correlate zscored means for each gene with each node in every simulation
        if sim_node is not None:
            simulated = simulated[simulated.index.get_level_values('gene') == sim_node]

        # todo: remove this inconsistency
        # Because of how the sim stats are labeled need to agument the conditions
        # and then remove
        conditions_labels = ["{}_mean".format(c) for c in self.training.values()]
        sim_means = simulated.loc[:, conditions_labels]
        sim_mean_z = sim_means.groupby(level='stat', axis=1).transform(stats.zscore, ddof=1).fillna(0)
        sim_mean_z.columns.set_levels([c.replace('_mean', "") for c in sim_mean_z.columns.levels[0]], level=0,
                                      inplace=True)
        corr = []
        for c in self.training.values():
            print('Computing pairwise for {}'.format(c))
            pcorr, p = pairwise_corr(sim_mean_z.loc[:, c], mean_z.loc[:, c], axis=1)
            corr.append(pcorr)
        print('Done')
        return (corr[0] + corr[1]) / 2

    def test(self):
        pass


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


def compile_sim(sim_dir, times, save_path=None, pp=True, **kwargs):
    # Initialize the results object
    gnr = GnwNetResults(sim_dir, **kwargs)

    print("Compiling simulation results. This could take a while")
    sim_results = gnr.compile_results(censor_times=times, save_intermediates=False,
                                      pp=pp)
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
    df = df[(df.score > s)] # & (df.p_value < p)]
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

def get_net_data(network, stim, directory, conditions):
    data_dir = '{}/{}/{}/'.format(directory, network, stim)
    results = [GnwSimResults(data_dir, network, c, sim_suffix='dream4_timeseries.tsv',
                          perturb_suffix="dream4_timeseries_perturbations.tsv").data for c in conditions]
    data = pd.concat(results).T.sort_index(axis=0).sort_index(axis=1)
    return data


def display_sim(network, stim, perturbation, times, directory, exp_condition='ko', ctrl_condition='wt', node=None):
    data_dir = '{}/{}/{}/'.format(directory, network, stim)
    network_structure = "{}/{}/{}_goldstandard_signed.tsv".format(directory, network, network)

    ctrl_gsr = GnwSimResults(data_dir, network, ctrl_condition, sim_suffix='dream4_timeseries.tsv',
                             perturb_suffix="dream4_timeseries_perturbations.tsv")
    exp_gsr = GnwSimResults(data_dir, network, exp_condition, sim_suffix='dream4_timeseries.tsv',
                            perturb_suffix="dream4_timeseries_perturbations.tsv")

    data = pd.concat([ctrl_gsr.data, exp_gsr.data]).T.sort_index(axis=0).sort_index(axis=1)
    if node:
        dep = DEPlot()
        idx = pd.IndexSlice
        dep.tsplot(data.loc[node, idx[:, :, perturbation, times]], subgroup='Time', no_fill_legend=True)
        plt.tight_layout()
        return
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