import os
from glob import iglob

import numpy as np
import pandas as pd
from scipy import stats


def augment_multiindex(df: pd.DataFrame, key, idx_name=''):
    if idx_name:
        df.index.name = idx_name
    df = df.set_index(key, append=True).sort_index()
    return df


def threshold(lfc_df, p_df, lfc_thresh=1, p_thresh=0.05):
    sign = np.sign(lfc_df)
    sign[np.abs(lfc_df) < lfc_thresh] = 0
    sign[p_df > p_thresh] = 0

    return sign.astype(int)


class GnwNetResults(object):
    """
    Load, manage, and analyze results from the GeneNetWeaver simulation results for one Network. Spawns multiple
    GnwSimResults and compares across conditions.

    This expects a specific directory structure based on GNWwrapper.py
    """
    def __init__(self, basepath, glob_seq="{}*/", experimental='ko', control='wt'):
        """
        :param basepath: path; directory where simulations results reside
        :param glob_seq: str; string containing a path specification
        :param experimental: str; identifier for the experimental condition
        :param control: str; identifier for the control condition
        :param censor_times: array-like; time points in the time series to keep for analysis (misnomer).
        :param sim_suffix: str; identifier for time series simulation data. See GnwSimResults
        :param perturb_suffix: str; identifier for perturbation data. See GnwSimResults
        """
        self.path = basepath
        self.sim_paths = iglob(glob_seq.format(self.path))
        self.experimental = experimental
        self.control = control

    def compile_results(self, censor_times=None, sim_suffix='dream4_timeseries.tsv',
                        perturb_suffix="dream4_timeseries_perturbations.tsv", save_intermediates=True):
        """
        Calculate log fold change and corresponding statistics across all networks

        :param experimental: str; identifier for the experimental condition
        :param control: str; identifier for the control condition
        :param sim_suffix: str; identifier for time series simulation data. See GnwSimResults
        :param perturb_suffix: str; identifier for perturbation data. See GnwSimResults
        :param censor_times: array-like; time points in the time series to keep for analysis (misnomer).
        :return:
        """
        results = pd.DataFrame()
        print('Compling results...')
        for ii, path in enumerate(self.sim_paths):
            id = os.path.basename(os.path.abspath(path))
            print(ii)

            # Get the data
            exp = GnwSimResults(path=path, sim_number=id, condition=self.experimental, sim_suffix=sim_suffix,
                                perturb_suffix=perturb_suffix, censor_times=censor_times)

            ctrl = GnwSimResults(path=path, sim_number=id, condition=self.control, sim_suffix=sim_suffix,
                                 perturb_suffix=perturb_suffix, censor_times=censor_times)

            # Get results and save them
            id_results = self.compare_conditions(exp.data, ctrl.data, id)
            if save_intermediates:
                id_results.to_csv(os.path.join(os.path.abspath(path), '{}_sim_stats.tsv'.format(id)), sep='\t')

            results = pd.concat([results, id_results])

        return results

    def compare_conditions(self, exp: pd.DataFrame, ctrl: pd.DataFrame, id, axis=0):
        """

        :param exp:
        :param ctrl:
        :return:
        """

        full = pd.concat([exp, ctrl]).groupby(level=['x_perturbation', 'Time'])
        results = full.apply(self.get_stats, 'ko', 'wt', axis).unstack()              # type: pd.DataFrame
        results = pd.concat([results], keys=[id], names=['id'])                     # type: pd.DataFrame

        # Move the gene names to the index
        results = results.stack(1)

        return results

    @staticmethod
    def get_stats(df, exp, ctrl, axis=0):
        """
        Calculate pvalues between conditions. Intended to use with "apply" from a pandas grouped dataframe
        :param df:
        :param exp:
        :param ctrl:
        :param axis:
        :return:
        """
        # todo: this throws a weird warning if the wrong axis is used
        p_val = stats.ttest_rel(df.loc[exp], df.loc[ctrl], axis=axis).pvalue
        exp_mean = np.mean(df.loc[exp], axis=axis)
        ctrl_mean = np.mean(df.loc[ctrl], axis=axis)
        lfc = np.log2(exp_mean/ctrl_mean)

        n = df.shape[1]
        multiindex = [np.array(['lfc']*n+['lfc_pvalue']*n+['{}_mean'.format(exp)]*n+['{}_mean'.format(ctrl)]*n),
                      np.array((df.columns.values.tolist()*4))]
        info = pd.Series(data=np.concatenate((lfc, p_val, exp_mean, ctrl_mean)), index=multiindex)
        info.index.names = ['stat', 'gene']

        return info


class GnwSimResults(object):
    """
    Load, manage, and analyze results from the GeneNetWeaver simulation results
    """
    def __init__(self, path, sim_number, condition, sim_suffix='dream4_timeseries.tsv',
                 perturb_suffix="dream4_timeseries_perturbations.tsv",censor_times=None):
        self.path = path
        self.id = sim_number
        self.condition = condition
        self.sim_suffix = sim_suffix
        self.perturb_suffix = perturb_suffix

        self.timeseries_data_file = None
        self.perturbation_data_file = None
        self.timeseries_data = None
        self.perturbation_data = None
        self.censor_times = censor_times
        self.data = None

        # Load and curate data
        self.load_data()
        self.annotated_data = self.annotate_data()
        self.set_data()

        # Calculate the relevant statistics
        self.sim_stats = self.calc_sim_stats()

    def load_data(self, ts_data=None, p_data=None):
        """
        Load the simulation data
        :param ts_data: path for the timeseries data
        :param p_data: path for the perturbation data
        :return:
        """

        if ts_data is None:
            ts_data = '{base}/{c}_sim/{id}_{c}_{s}'.format(base=self.path, c=self.condition, id=self.id,
                                                           s=self.sim_suffix)

        if p_data is None:
            p_data = '{base}/{c}_sim/{id}_{c}_{s}'.format(base=self.path, c=self.condition, id=self.id,
                                                          s=self.perturb_suffix)

        self.timeseries_data_file = os.path.abspath(ts_data)
        self.perturbation_data_file = os.path.abspath(p_data)

        self.timeseries_data = pd.read_csv(self.timeseries_data_file, sep='\t')
        self.perturbation_data = pd.read_csv(self.perturbation_data_file, sep='\t')

        return

    def annotate_data(self):
        times = sorted(list(set(self.timeseries_data['Time'].values)))
        n_timeseries = len(self.timeseries_data) / len(times)
        perturbations = np.array(sorted(list(set(self.perturbation_data['x']))))

        # For safety
        if not n_timeseries.is_integer():
            raise ValueError('Number of time points for each replicate is not the same')

        assert n_timeseries == len(self.perturbation_data)

        # Assign replicate number to each row of the perturbations
        p_rep_list = np.ceil((self.perturbation_data.index.values + 1) / len(perturbations)).astype(int)

        # For each row in the time series, calculate the perturbation simulation index
        ts_p_index = np.ceil((self.timeseries_data.index.values + 1) / len(times)).astype(int) - 1

        # Assign a replicate number to each row of the time series based on perturbation
        ts_rep_list = p_rep_list[ts_p_index]

        # Get the actual perturbation value used in the simulation for each row of the time series
        ts_p_list = self.perturbation_data.loc[list(ts_p_index), 'x'].values

        # Add the annotations to the dataframe
        annotated_data = self.timeseries_data.copy()
        annotated_data['x_perturbation'] = ts_p_list
        annotated_data['rep'] = ts_rep_list
        annotated_data['condition'] = self.condition

        annotated_data.set_index(['condition', 'rep', 'x_perturbation', 'Time'], inplace=True)
        annotated_data.sort_index(inplace=True)

        return annotated_data

    def censor_data(self, data, times=None):
        """
        Only analyze certain timepoints
        :param times: list of times to keep in the analysis
        :param data:
        :return:
        """
        if times is None:
            times = self.censor_times
        idx = pd.IndexSlice
        censored = data.loc[idx[:, :, :, times], :].copy()
        return censored

    def set_data(self):
        """
        Set the data used for analysis and censor timepoints if necessary
        :return:
        """
        data = self.annotated_data.copy()
        if self.censor_times is not None:
            data = self.censor_data(data, self.censor_times)

        self.data = data

        return

    def calc_sim_stats(self, grouping=('x_perturbation', 'Time'), calc=('mean', 'std')):
        """
        Calculate stats across different dimensions
        :param grouping: list-like; strings in the multiindex that define the groupby object. Default is
            'x_perturbation' and 'Time'.
        :param calc: list-like; str for pandas builtin or func that can be used on each series defined in the grouping.
            Defaults are mean and std.
        :return:
        """
        # Make groupby object for efficient pandas calculations
        g = self.data.groupby(level=grouping)

        # Aggregate the statistics.
        sim_stats = (g.agg(calc))

        return sim_stats