import multiprocessing as mp
import os
import sys
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from pydiffexp import DEAnalysis, DEPlot, pairwise_corr
from pydiffexp.gnw import GnwNetResults, GnwSimResults, draw_results, get_graph
from scipy import stats
from sklearn.metrics import mean_squared_error as mse


class DynamicDifferentialExpression(object):
    """
    Coordinate training and testing of DDE models
    """
    def __init__(self, directory, p=0.05):
        self.dir = None
        self.project = None
        self.p = p
        self.training = {}          # type: dict
        self.test = []
        self.estimators = None      # type: pd.core.groupby.DataFrameGroupBy
        self.dea = None             # type: DEAnalysis
        self.sim_stats = None       # type: pd.DataFrame
        self.corr = None            # type: pd.DataFrame
        self.sim_scores = None      # type: pd.DataFrame
        self.match = None           # type: pd.DataFrame
        self.ddegs = None           # type: pd.DataFrame
        self.ddeg_contrast = None   # type: pd.DataFrame
        self.times = None           # type: list
        self.sim_dea = None         # type: DEAnalysis

        # There is a lot of intermediary data that can be saved to make rerunning the analysis easier
        self.set_save_directory(directory)
        self.dea_path = None
        self.scores_path = None
        self.corr_path = None

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

    def score(self, prefix, exp, ctrl, sim_predictions=None, sim_filter=None,
              f=None, reduced_set=True, plot=False):

        if f is None:
            f = mse

        contrast = "{}-{}".format(exp, ctrl)

        true_der = self.dea.results[contrast]
        true_lfc = true_der.top_table().iloc[:, :len(self.times)]

        if sim_predictions is None:
            sim_path = os.path.join(self.dir,
                                    '{}_{}_sim_stats.pkl'.format(prefix,
                                                                 contrast))

            # Get the predicted values
            sim_predictions = self.load_sim_stats(sim_path, self.times, exp, ctrl,
                                                  **sim_filter)
        pred_der = self.sim_dea.results[contrast]
        pred_lfc = pred_der.top_table().iloc[:, :len(self.times)]
        pred_lfc.index = pred_lfc.index.astype(int)
        train_lfc = self.sim_dea.results['ko-wt'].coefficients.abs().mean(axis=1)
        train_lfc.index = train_lfc.index.astype(int)

        # Reduce the number of genes to score against, for speed purposes
        if reduced_set:
            true_lfc = true_lfc.loc[list(set(self.match['train_gene']))]

        # Create a dictionary of each simulations prediction to each matched gene
        # This is the distribution of the null model for randomly chosen models
        # gene_err_dist_dict = {ii: [f(pwlfc, twlfc) for pwlfc in pred_lfc.values]
        #                       for ii, twlfc in true_lfc.iterrows()}
        # #
        # gene_to_model_error = pd.DataFrame.from_dict(gene_err_dist_dict,
        #                                              orient='index')
        # #
        # # # Set the columns to match the models
        # gene_to_model_error.columns = pred_lfc.index
        # gene_to_model_error.to_pickle('gtme.pkl')
        gene_to_model_error = pd.read_pickle('gtme.pkl')

        # Calculate null model
        null_stats = self.estimators.apply(self.sample_stats, gene_to_model_error,
                                           true_lfc, pred_lfc, true_der, train_lfc)

        # Combine the stats together
        test_stats = pd.concat([self.ddegs, null_stats], axis=1).dropna()

        # Add known cluster info in
        test_stats['ki_cluster'] = true_der.score_clustering().loc[test_stats.index, 'Cluster']

        if plot:
            self.plot_results(test_stats)

        return test_stats

    @staticmethod
    def plot_results(x):
        print(x)
        print(np.median(x.grouped_diff), np.mean(x.grouped_diff), stats.wilcoxon(x.grouped_diff).pvalue/2)
        print(np.median(x.avg_diff), np.mean(x.avg_diff), stats.wilcoxon(x.avg_diff).pvalue/2)
        melted = pd.melt(x, id_vars=x.columns[:4], value_vars=x.columns[4:10], var_name='stat')
        plt.figure(figsize=(3, 5))
        ax = sns.boxplot(data=melted, x='stat', y='value', notch=True, showfliers=False, width=0.5)
        # sns.swarmplot(data=melted, x='stat', y='value', color='black')
        # ax.set(xticklabels=['mean', 'random_mean','mean_diff', 'grouped', 'random_grouped', 'grouped_diff'])
        plt.xticks(rotation=90)
        plt.xlabel("")
        plt.ylabel("Prediction MSE")
        plt.tight_layout()
        plt.show()
        sys.exit()

    @staticmethod
    def moderate_lfc(df):
        avg_lfc = df.median()
        if len(df) > 2:
            p = avg_lfc * (1-stats.ttest_1samp(df, 0).pvalue)
        else:
            p = avg_lfc
        return p

    def sample_stats(self, df, dist_dict, true_lfc, pred_lfc, true_der, trainlfc,
                     resamples=100, err=mse):
        # Test filter
        gene = df.name

        # preddev = pred_lfc.abs().mean(axis=1)
        # df['preddev'] = [preddev.loc[ii] for ii in df['index']]
        # df = df[df.preddev < 1].copy()

        # For readability

        models = df['index'].values.astype(int)
        n = len(df)
        # if n < 1:
        #     return

        # Get the true log fold change for this dataframe
        test = true_lfc.loc[gene]
        # t = test*(1-true_der.p_value.loc[gene])

        # Get the distribution of errors for all models to this gene
        e_dist = dist_dict.loc[gene]
        
        # Calculate prediction error
        elfc = pd.concat([e_dist.loc[models], trainlfc.loc[models]], keys=['error', 'sumlfc'], axis=1)

        # Group the models log fold change predictions together for each time point
        # then calculate the error of the 'averaged' model
        grouped_prediction = pred_lfc.loc[models].median()
        # p = self.moderate_lfc(pred_lfc.loc[models])
        grouped_error = err(test, grouped_prediction)

        group_dev = grouped_prediction.abs().sum()

        # Calculate a predicted cluster

        # n = self.moderate_lfc(pred_lfc)
        n = pred_lfc.median()
        nonzero = [stats.ttest_1samp(pred_lfc.loc[models, t], 0).pvalue < 0.05 for t in pred_lfc.columns]
        grouped_cluster = (np.sign(grouped_prediction)*nonzero).astype(int)
        grouped_cluster = str(tuple(grouped_cluster.values.tolist()))

        # Average error of each model to the true values
        avg_error = e_dist.loc[models].median()

        # The dimensions must be consistent
        assert dist_dict.shape[1] == pred_lfc.shape[0]

        # Get random sample indices
        # rs = np.random.randint(0, len(pred_lfc), (resamples, len(df)))

        # Calculate null models
        # rs_avg_error = [np.median(e_dist.iloc[r]) for r in rs]
        # rg_lfc_error = [err(t, self.moderate_lfc(pred_lfc.loc[r])) for r in rs]

        # Calculate the average across all the random samples
        # The average of the medians should be close to the true median
        rs_median = e_dist.median()
        rg_median = err(test, n)
        # rg_median = np.median(rg_lfc_error)

        # Error if all log fold change values are assumed to be zero
        all_zeros = err(test, np.zeros((len(test))))
        magnitude = err(grouped_prediction, np.zeros(len(grouped_prediction)))

        # Return a series of statistics
        s_labels = ['n', 'grouped_mag','grouped_e', 'random_grouped_e', 'grouped_diff',
                    'avg_e', 'random_avg_e', 'avg_diff', 'all_zeros', 'abs_dev', 'group_dev',
                    'group_cluster']
        s_values = [len(df), magnitude, grouped_error, rg_median, rg_median-grouped_error,
                    avg_error, rs_median, rs_median-avg_error, all_zeros, test.abs().mean(),
                    group_dev, grouped_cluster]
        s = pd.Series(s_values, index=s_labels)
        return s

    def random_sample(self, df: pd.DataFrame):
        n_estimators = len(df)
        possible_estimators = self.sim_stats.index.values
        random_estimators = np.random.choice(possible_estimators, n_estimators)
        df.index = pd.MultiIndex.from_tuples(random_estimators,
                                             names=df.index.names)

        return df

    def predict(self, test, estimators=None, ctrl=None):
        """
        Calculate predictions from trained estimators
        :param test:
        :param estimators:
        :param ctrl:
        :return:
        """
        # todo: it would be ideal if this could rapidly query the estimators with flexible time sampling
        self.set_test_conditions(test)
        if ctrl is None:
            ctrl = self.training['control']

        contrast = '{}-{}'.format(test, ctrl)

        # Set default.
        if estimators is None:
            estimators = self.estimators

        # Calculate estimator predictions
        prediction = estimators.apply(self._e_predict, contrast=contrast,
                                      ctrl=ctrl)

        return prediction

    def score_prediction(self, c, prediction, f=None):
        if f is None:
            f = mse
        true_data = self.dea.data.loc[prediction.index, c]
        timeseries_mean = true_data.groupby(level='time', axis=1)
        true_value = timeseries_mean.mean()
        error = true_value.apply(lambda x: f(x, prediction.loc[x.name]), axis=1)

        return error

    def _e_predict(self, df, contrast, ctrl):
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
        sim_predictors = df['net'].astype(str)
        est_lfc = self.sim_dea.results[contrast].coefficients.loc[sim_predictors]
        est_lfc.columns = baseline.index
        pred = baseline + est_lfc

        return pred

    def set_paths(self, prefix, contrast):
        """
        Set paths for intermediate data
        :param prefix:
        :param contrast:
        :return:
        """
        prepend = '{}_{}'.format(prefix, contrast)
        self.corr_path = os.path.join(self.dir, '{}_data_to_sim_corr.pkl'.format(prepend))
        return

    def fit_data(self, data, **kwargs):
        # Default expects count data and no additional log2
        kwargs.setdefault('counts', True)
        kwargs.setdefault('log2', False)

        self.dea = self.fit_dea(self.dea, data, **kwargs)
        self.times = self.dea.times

    def fit_sim(self, data, **kwargs):
        kwargs.setdefault('counts', False)
        kwargs.setdefault('log2', True)
        self.sim_dea = self.fit_dea(self.sim_dea, data, **kwargs)

    @staticmethod
    def fit_dea(dea, data=None, override=False, **kwargs):
        """

        :param data:
        :param default_contrast:
        :param kwargs:
        :return:
        """

        if dea is None or override:
            # Set defaults
            kwargs.setdefault('reference_labels', ['condition', 'time'])
            kwargs.setdefault('index_names', ['condition', 'replicate', 'time'])

            # Make the dea object and fit it
            new_dea = DEAnalysis(data, **kwargs)
            new_dea.fit_contrasts(new_dea.default_contrasts, status=True)

        else:
            new_dea = dea

        return new_dea

    @staticmethod
    def match_to_gene(x, y, correlation, unique_net=True):

        # Match the dtype
        y.index = y.index.astype(correlation.columns.dtype)

        match = pd.DataFrame()

        # For each row in the actual data, match all networks in y with the same
        # cluster value
        for gene, row in x.iterrows():
            candidate_nets = y.loc[y.Cluster == row.Cluster]
            cur_corr = correlation.loc[gene, candidate_nets.index.values]
            cur_corr.name = 'pearson_r'
            ranking = pd.concat([candidate_nets, cur_corr], axis=1)
            ranking['mean'] = (ranking['cscore'] + ranking['pearson_r']) / 2

            # Remove same network ids that are just different perturbations
            if unique_net:
                sorted_ranking = ranking.sort_values('mean', ascending=False)
                ranking = sorted_ranking[~sorted_ranking.index.get_level_values('id').duplicated(keep='first')].copy()

            # Add the gene name that is matched
            ranking['train_gene'] = gene

            # Add it the dataframe
            ranking.index.name = 'net'
            match = pd.concat([match, ranking.reset_index()], ignore_index=True)

        match.sort_values('mean', ascending=False, inplace=True)
        match = match[(match['cscore'] > 0) & (match['pearson_r'] > 0)]

        return match

    def train(self, project, data, sim_data, override=False, exp='ko', ctrl='wt',
              data_kwargs=None, sim_kwargs=None):
        """
        Train DDE estimators

        :param project:
        :param data:
        :param sim_data:
        :param override:
        :param exp:
        :param ctrl:
        :param data_kwargs:
        :param sim_kwargs:
        :return:
        """

        # Set conditions
        contrast = "{}-{}".format(exp, ctrl)
        self.set_training_conditions(exp, ctrl)
        self.project = project
        self.ddeg_contrast = contrast

        # Define paths to save or read pickles from
        self.set_paths(project, contrast)

        # Fit the expression data
        if data_kwargs is None:
            data_kwargs = {}
        self.fit_data(data, override=override, **data_kwargs)

        # Fit the simulation data
        if sim_kwargs is None:
            sim_kwargs = {}
        self.fit_sim(sim_data, override=override, **sim_kwargs)

        # Get dDEGs
        ddegs = self.dea.results[contrast].get_dDegs()

        # Also filter out genes that don't pass the basic pairwise test (not DEG)
        degs = self.dea.results[contrast].top_table(p=self.p)
        ddegs = ddegs.loc[set(ddegs.index).intersection(degs.index)].copy()

        # Get the data needed for the correlation
        filtered_data = self.dea.data.loc[:, contrast.split('-')]
        filtered_sim = self.sim_dea.data.loc[:, contrast.split('-')]

        # Correlate the mean trajectories
        corr = self.correlate(filtered_data, filtered_sim, override=override)

        # Match the genes to simulation networks
        sim_scores = self.sim_dea.results[contrast].cluster_scores
        match = self.match_to_gene(ddegs, sim_scores, corr, unique_net=False)

        self.estimators = match.groupby('train_gene')
        self.match = match
        self.ddegs = set(ddegs.index)

        return match

    def correlate(self, exp: pd.DataFrame, sim: pd.DataFrame, sim_node=None,
                  override=False):
        try:
            if override:
                raise ValueError('Override to retrain')
            corr = pd.read_pickle(self.corr_path)
        except (FileNotFoundError, ValueError):

            # Get group means and zscore
            gene_mean = exp.groupby(level=['condition', 'time'], axis=1).mean()
            mean_z = gene_mean.groupby(level='condition', axis=1).transform(stats.zscore, ddof=1).fillna(0)

            # Correlate zscored means for each gene with each node in every simulation
            if sim_node is not None:
                sim = sim[sim.index.get_level_values('gene') == sim_node]

            sim_mean = sim.groupby(level=['condition', 'time'], axis=1).mean()
            sim_mean_z = sim_mean.groupby(level='condition', axis=1).transform(stats.zscore, ddof=1).fillna(0)

            all_corr = []
            for c in self.training.values():
                print('Computing pairwise for {}...'.format(c), end=' ', flush=True)
                pcorr, p = pairwise_corr(sim_mean_z.loc[:, c], mean_z.loc[:, c], axis=1)
                all_corr.append(pcorr)
                print('DONE')
            corr = (all_corr[0] + all_corr[1]) / 2

        return corr

    def to_pickle(self, path=None, force_save=False):
        # Note, this is taken directly from pandas generic.py which defines the method in class NDFrame
        """
        Pickle (serialize) object to input file path

        Parameters
        ----------
        path : string
            File path
        """
        should_pickle = True

        # Set the save directory
        if path is None:
            fname = "{}_{}_dde.pkl".format(self.project, self.ddeg_contrast)
            path = os.path.join(self.dir, fname)

        if not os.path.exists(os.path.dirname(path)):
            sys.exit('The directory entered to save the pickle to, "%s", does not exist' % os.path.dirname(path))

        # If the pickle path exists, ask if the user wants to save over it
        if os.path.isfile(path) and not force_save:
            print("Pickle file to save: ", path)
            answer = input('The proposed pickle file already exists. Would you like to replace it [y/n]?')
            if answer != 'y':
                should_pickle = False
                if answer != 'n':
                    warnings.warn('Invalid answer')

        if should_pickle:
            print("Pickling object to %s" % os.path.abspath(path))
            pd.to_pickle(self, path)
        else:
            sys.exit("Object not pickled."
                     "\nTo save object please rerun with a different file path or choose to rewrite")

        return


def compile_sim(sim_dir, times, save_path=None, pp=True, **kwargs):
    # Initialize the results object
    gnr = GnwNetResults(sim_dir, **kwargs)

    print("Compiling simulation results. This could take a while")
    sim_results = gnr.compile_results(censor_times=times, save_intermediates=False,
                                      pp=pp)
    if save_path is not None:
        sim_results.to_pickle(save_path)
    return sim_results


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