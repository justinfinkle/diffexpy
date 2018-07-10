import ast
import itertools
import os
import sys
import warnings
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from natsort import natsorted
from pydiffexp.utils import multiindex_helpers as mi
from pydiffexp.utils import rpy2_helpers as rh
from pydiffexp.utils.utils import int_or_float, grepl
from rpy2.robjects.packages import importr

# Activate conversion
rpy2.robjects.numpy2ri.activate()

# Load R packages
limma = importr('limma')
stats = importr('stats')

# Set null variable
null = robjects.r("NULL")


def cluster_discrete(df) -> pd.DataFrame:
    """
    Cluster trajectories into tuples that can be easily counted
    :param df: DataFrame; expected to come from decide_tests
    :return:
    """
    clusters = [str(tuple(gene)) for gene in df.values]
    cluster_df = pd.DataFrame(clusters, index=df.index, columns=['Cluster'])

    return cluster_df


def cluster_to_array(cluster: str):
    """
    Converts a string of DE cluster values to an integer array
    :param cluster: str: expected form (int, int, int,...)
    :return:
    """
    return np.array([int(s) for s in cluster.strip('())').split(',')])


def get_scores(grouped_df, de_df, weighted_df):
    """
    Score how well each trajectory fits the assigned cluster
    :param grouped_df: pandas grouped dataframe
    :param de_df: differential expression dataframe
    :param weighted_df: weighted dataframe
    :return: df
    """
    # If there is only one group it needs to be setup differently
    if len(grouped_df.groups.keys()) == 1:
        # Extract group as a dataframe
        single_cluster = list(grouped_df.groups.keys())[0]
        g = grouped_df.get_group(single_cluster)
        g.name = single_cluster
        scores = group_scores(g, de_df, weighted_df)

        # Format to match expected output of apply
        scores = scores.reset_index()
        scores.insert(0, 'Cluster', single_cluster)
        scores.set_index(['Cluster', 'index'], inplace=True)
    else:
        scores = pd.DataFrame(grouped_df.apply(group_scores, de_df, weighted_df))
    if 'gene' not in scores.index.names:
        scores.index.set_names('gene', level=1, inplace=True)
    scores = scores.reset_index().sort_values(['Cluster', 'score'], ascending=[False, False]).set_index('gene')
    scores.fillna(0, inplace=True)
    return scores


def group_scores(cluster, de: pd.DataFrame, weighted_de: pd.DataFrame):
    """
    Score how well each trajectory matches the cluster
    :param cluster:
    :param de:
    :param weighted_de:
    :return: series
    """
    # Get the scores for each trajectory based on the assumed cluster
    expected = np.array(eval(cluster.name))
    clus_de = de.loc[cluster.index]
    clus_wde = weighted_de.loc[cluster.index]
    diff = clus_de - clus_wde
    correct_de = ~np.abs(np.sign(clus_wde).values - expected).astype(bool)
    penalty = (correct_de*diff + (~correct_de)*clus_wde).abs().sum(axis=1)
    reward = (correct_de*clus_wde + (~correct_de)*diff).abs().sum(axis=1)

    # scores = np.abs(clus_wde).values * penalty + np.abs(clus_wde).values * (penalty == 0)  # type: np.ndarray

    # Calculate fraction of the lfc that was retained
    score_frac = (reward-penalty)/clus_de.abs().sum(axis=1)
    # score_frac = np.sum(scores, axis=1)/np.sum(np.abs(de_df.loc[cluster.index]).values, axis=1)
    # score_frac = 1 - (de.loc[cluster.index] - clus_wde).abs().sum(axis=1) / de.loc[cluster.index].abs().sum(axis=1)
    score_frac.name = 'score'

    return score_frac
    # return pd.Series(data=score_frac, index=cluster.index, name='score')


class MArrayLM(object):
    """
    Class to wrap MArrayLM from R. Makes data more easily accessible

    Note: This will probably never be directly instantiated because DEResults inherits from and extends upon
    this base class

    """
    def __init__(self, obj):
        """

        :param obj:
        """
        # Store the original object
        self.robj = obj                 # type: robj.vectors.ListVector

        # Initialize expected attributes. See R documentation on MArrayLM for more details on attributes
        self.Amean = None               # type: np.ndarray
        self.F = None                   # type: np.ndarray
        self.F_p_value = None           # type: np.ndarray
        self.assign = None              # type: np.ndarray
        self.coefficients = None        # type: pd.DataFrame
        self.contrasts = None           # type: pd.DataFrame
        self.cov_coefficients = None    # type: pd.DataFrame
        self.design = None              # type: pd.DataFrame
        self.df_prior = None            # type: float
        self.df_residual = None         # type: np.ndarray
        self.df_total = None            # type: np.ndarray
        self.lods = None                # type: pd.DataFrame
        self.method = None              # type: str
        self.p_value = None             # type: pd.DataFrame
        self.proportion = None          # type: float
        self.qr = None                  # type: dict
        self.rank = None                # type: int
        self.s2_post = None             # type: np.ndarray
        self.s2_prior = None            # type: float
        self.sigma = None               # type: np.ndarray
        self.stdev_unscaled = None      # type: pd.DataFrame
        self.t = None                   # type: pd.DataFrame
        self.var_prior = None           # type: float

        # Unpack the values
        self.unpack()
        self.contrast_list = self.contrasts.columns.values

    def unpack(self):
        """
        Unpack the MArrayLM object (rpy2 listvector) into an object.
        """
        # Unpack the list vector object
        data = rh.unpack_r_listvector(self.robj)

        # Store the values into attributes
        for k, v in data.items():
            setattr(self, k, v)


class DEResults(MArrayLM):
    """
    Class intended to organize results from differential expression analysis in easier fashion
    """
    def __init__(self, fit, name=None, fit_type=None):
        # Call super
        super(DEResults, self).__init__(fit)

        self.name = name                                                                # type: str
        self.fit_type = fit_type                                                        # type: str
        self.continuous_kwargs = {'coef': null, "number": 10, 'genelist': null,
                                  "adjust_method": "BH", "sort_by": "B", "resort_by": null,
                                  "p_value": 0.05, "lfc": 0, "confint": False}          # type: dict
        self.discrete_kwargs = {'method': 'separate', 'adjust_method': 'BH',
                                'p_value': 0.05, 'lfc': 0}                              # type: dict
        self.continuous = self.top_table(**self.continuous_kwargs)                      # type: pd.DataFrame
        self.discrete = self.decide_tests(**self.discrete_kwargs)                       # type: pd.DataFrame
        self.discrete_clusters = cluster_discrete(self.discrete)                        # type: pd.DataFrame
        self.cluster_count = self.count_clusters(self.discrete_clusters)                # type: pd.DataFrame
        self.all_results = self.aggregate_results()                                     # type: pd.DataFrame

    def aggregate_results(self):
        """
        Make a hierarchical dataframe to include all results
        :return:
        """

        # Make the multiindex
        discrete = pd.concat([self.discrete_clusters, self.discrete], axis=1)           # type: pd.DataFrame
        idx_array = [['discrete']*discrete.shape[1]+['continuous']*self.continuous.shape[1],
                     discrete.columns.values.tolist()+self.continuous.columns.values.tolist()]
        idx = pd.MultiIndex.from_arrays(idx_array, names=['dtype', 'label'])
        hierarchical = pd.concat([discrete, self.continuous], axis=1)
        hierarchical.columns = idx
        return hierarchical

    def count_clusters(self, df, column='Cluster') -> pd.DataFrame:
        """
        Count the clusters in each category
        :param df:
        :param column:
        :return:
        """
        n_timepoints = self.contrasts.shape[1]
        zeros = str(tuple([0] * n_timepoints))
        counts = Counter(df[column].tolist())
        del counts[zeros]

        cluster_count = pd.DataFrame.from_dict(counts, orient='index')

        # There may not be any clusters
        try:
            cluster_count.columns = ['Count']

            # Calculate difference from expectation of evenly distributed bins
            expected_clusters = 3 ** n_timepoints
            num_clusters = len(counts)
            n_genes = sum(counts.values())
            expected_per_cluster = round(n_genes/expected_clusters)

            cluster_count['Diff from Expectation'] = np.abs(cluster_count['Count']-expected_per_cluster)
        except ValueError:
            # Return the empty dataframe
            pass

        cluster_count.sort_values('Count', ascending=False, inplace=True)

        return cluster_count

    def top_table(self, use_fstat=None, p=1, n='inf', **kwargs) -> pd.DataFrame:
        """
        Print top_table of differential expression analysis
        :param fit: MArrayLM; a fit object created by DEAnalysis
        :param use_fstat: bool; select genes using F-statistic. Useful if testing significance for multiple contrasts,
        such as a time series
        :param p float; cutoff for significant top_table. Default is 0.05. If np.inf, then no cutoff is applied
        :param n int or 'inf'; number of significant top_table to include in output. Default is 'inf' which includes all
        top_table passing the threshold
        :param kwargs: additional arguments to pass to topTable. see topTable documentation in R for more details.
        :return:
        """

        # Update kwargs with commonly used ones provided in this API
        kwargs = dict(kwargs, p_value=p, number=n)

        # Use fstat if multiple contrasts supplied
        if use_fstat is None:
            use_fstat = False if (isinstance(self.contrast_list, str) or
                                  (isinstance(self.contrast_list, list) and len(self.contrast_list) == 1)) else True

        if 'coef' in kwargs.keys() and kwargs['coef'] != null:
            single_contrast = True
            if use_fstat:
                warnings.warn('Cannot specify use_fstat=True when a specifiying a value for "coef"\n'
                              'use_fstat will be set to False')
                use_fstat = False
        else:
            single_contrast = False

        if use_fstat:
            # Modify parameters for use with topTableF
            kwargs['sort_by'] = 'F'

            # Drop values that won't be used by topTableF
            for k in ['coef', 'resort_by', 'confint']:
                if k in kwargs.keys():
                    del kwargs[k]
            table = limma.topTableF(self.robj, **kwargs)
        else:
            table = limma.topTable(self.robj, **kwargs)

        # Add use_fstat
        kwargs = dict(kwargs, use_fstat=use_fstat)
        self.continuous_kwargs = kwargs

        df = rh.rvect_to_py(table)

        # Rename the column and add the negative log10 values
        df.rename(columns={'adj.P.Val': 'adj_pval', 'P.Value': 'pval'}, inplace=True)  # Remove expected periods
        df['-log10p'] = -np.log10(df['adj_pval'])

        if not single_contrast:
            df_cols = df.columns.values.tolist()
            df_cols[:len(self.contrast_list)] = self.contrast_list
            df.columns = df_cols

        return df

    def decide_tests(self, m='global', **kwargs) -> pd.DataFrame:
        """
        Determine if each gene is significantly differentially expressed based on criteria. Returns discrete values.

        :param m: str; Method used for multiple hypothesis testing. See R documentation for more details.
        :param kwargs: Additional keyword arguments available in R.
        :return: DataFrame; 1 for overexpressed, -1 for underexpressed, 0 if not significantly different.
        """
        # Update kwargs with commonly used ones provided in this API
        kwargs = dict(kwargs, method=m)

        # Run decide tests
        decide = limma.decideTests(self.robj, **kwargs)

        self.discrete_kwargs = kwargs

        # Convert to dataframe
        df = rh.rvect_to_py(decide).astype(int)
        return df

    def score_clustering(self, grouped=None, ind_p=0.05):
        # Calculate the weighted log fold change as lfc*(1-pvalue) at each time point
        weighted_lfc = (1 - self.p_value) * self.continuous.loc[self.p_value.index, self.p_value.columns]

        # Group genes by clusters
        if grouped is None:
            grouped = cluster_discrete(self.decide_tests(p_value=ind_p)).groupby('Cluster')

        # Score the clustering
        scores = get_scores(grouped, self.continuous.loc[:, self.p_value.columns], weighted_lfc).sort_index()
        scores['score'] = scores['score']*(1-self.continuous['adj_pval']).sort_index().values
        scores.sort_values('score', ascending=False, inplace=True)

        return scores


class DEAnalysis(object):
    """
    An object that does differential expression analysis with time course data
    """

    def __init__(self, df=None, index_names=None, split_str='_', time='time', condition='condition',
                 replicate='replicate', reference_labels=None, voom=False, log2=True):
        """

        :param df:
        :param index_names:
        :param split_str:
        :param time:
        :param condition:
        :param replicate:
        :param reference_labels:
        """

        self.raw = None                     # type: pd.DataFrame
        self.data = None                    # type: pd.DataFrame
        self.labels = None
        self.times = None                   # type: list
        self.conditions = None              # type: list
        self.replicates = None              # type: list
        self.voom = voom                    # type: bool
        self.voom_results = None
        self.default_contrasts = None
        self.timeseries = False             # type: bool
        self.samples = None                 # type: list
        self.experiment_summary = None      # type: pd.DataFrame
        self.design = None                  # type: robjects.vectors.Matrix
        self.data_matrix = None             # type: robjects.vectors.Matrix
        self.contrast_robj = None           # type: robjects.vectors.Matrix
        self.fit = None                     # type: dict
        self.results = {}                   # type: Dict[str, DEResults]
        self.contrast_dict = {}             # type: dict
        self.decide = None                  # type: pd.DataFrame
        self.db = None                      # type: pd.DataFrame
        self.log2 = log2                    # type: bool

        if df is not None:
            # Set the data
            self._set_data(df, index_names=index_names, split_str=split_str, reference_labels=reference_labels,
                           voom=voom, log2=log2)

            # Determine if data is timeseries
            self.times, self.timeseries = self._is_timeseries(time_var=time)

            # Determine conditions and replicates of experiment
            self.conditions = sorted(list(set(self.experiment_summary[condition])))
            self.replicates = sorted(list(set(self.experiment_summary[replicate])))

            # Set default contrasts
            self.default_contrasts = self.possible_contrasts()

    def _set_data(self, df, index_names=None, split_str='_', reference_labels=None, voom=False, log2=True):
        """
        Set the data for the DEAnalysis object
        :param df: DataFrame; 
        :param index_names: list-like;
        :param split_str: string;
        :param reference_labels: list-like;
        :param voom: bool; Voom the data for fitting. True if using counts data from RNA-seq. False if using microarray
        :return: 
        """
        # Check for a multiindex or try making one
        multiindex = mi.is_multiindex(df)
        h_df = None
        if sum(multiindex) == 0:
            h_df = mi.make_multiindex(df, index_names=index_names, split_str=split_str)
        else:
            if multiindex[1]:
                h_df = df.copy()
            elif multiindex[0] and not multiindex[1]:  # Second part is probably redundant
                h_df = df.T
                warnings.warn('DataFrame transposed. Multiindex is along columns.')

            # Double check multiindex
            multiindex = mi.is_multiindex(h_df)
            if sum(multiindex) == 0:
                raise ValueError('No valid multiindex was found, and one could not be created')

        # Sort multiindex
        try:
            h_df.sort_index(axis=1, inplace=True)
        except TypeError:
            # If mixed types within a level, the sort won't work
            pass
        self.raw = h_df

        # Summarize the data and make data objects for R
        self.experiment_summary = self.get_experiment_summary(reference_labels=reference_labels)
        self.design = self._make_model_matrix()
        self.data_matrix = self._make_data_matrix(voom=voom, log2=log2)

    def _is_timeseries(self, time_var=None):
        """
        Decide if data is timeseries or not. If it is, return the unique times
        :param time_var: str; Column name of the time variable
        :return: 
        """
        times = []
        is_timeseries = False

        # Future autodetect - find timevariable
        # print(grepl('TIME', self.experiment_summary.columns.str.upper()))

        if time_var is not None:
            # Get unique values of times
            times = sorted(map(int_or_float, list(set(self.experiment_summary[time_var]))))

            # If there is more than one time value, dataset is a timeseries
            if len(times) > 1:
                is_timeseries = True

        return times, is_timeseries

    def get_experiment_summary(self, reference_labels=None):
        """
        Summarize the experiment details in a data frame
        :return:
        """
        index = self.raw.columns
        summary_df = pd.DataFrame()
        for ii, name in enumerate(index.names):
            summary_df[name] = index.levels[ii].values[index.labels[ii]].astype(str)
        if reference_labels is not None:
            summary_df['sample_id'], self.labels, self.samples = self.make_sample_ids(summary_df,
                                                                                      reference_labels=reference_labels)
        else:
            warnings.warn('Sample IDs and labels not set because no reference labels supplied. R data matrix and '
                          'contrasts cannot be created without sample IDs. Setting sample labels to integers')
            self.labels = list(map(lambda x: 'x%i' % x, range(len(summary_df))))
        return summary_df

    @staticmethod
    def _split_samples(x):
        find_str = grepl('-', x)
        if len(find_str) > 0:
            x = [sample.replace('(', "").replace(')', '').split('-') for sample in x]
            x = set().union(*x)
        else:
            x = set(x)
        return x

    def _contrast(self, v1, v2, fit_type, join_str='-'):
        contrast = {'contrasts': list(map(join_str.join, zip(v1, v2))),
                    'fit_type': fit_type,
                    'samples': set(self._split_samples(v1)).union(self._split_samples(v2))}
        return contrast

    def possible_contrasts(self, p_idx=0):
        """
        Make a list of expected contrasts based on times and conditions
        :param p_idx: int; the index of the time sample when the perturbation was applied
        :return:
        """

        '''
        Contrast classes - DE, TS, AR, TS-DE, DE-TS, AR-DE, DE-AR
        DE: Differential Expression
        TS: Time Series
        AR: Autoregression
        '''

        if self.timeseries:
            # Labels to append to keys to signify contrast type
            ts_str = '_ts'
            ar_str = '_ar'

            # List of condition pairs
            condition_combos = list(itertools.combinations_with_replacement(self.conditions, 2))
            contrasts = {}

            # Build basic contrasts
            for c in condition_combos:
                # Time series
                if c[0] == c[1]:
                    x = c[0]
                    samples = grepl(x, self.samples)
                    contrasts[str(x) + ts_str] = self._contrast(samples[1:], samples[:-1], 'TS')

                    # Autoregression
                    ar_samples = [samples[p_idx]] * (len(samples) - 1)
                    contrasts[str(x) + ar_str] = self._contrast(samples[1:], ar_samples, 'AR')

                # Static
                else:
                    contrasts['-'.join(c)] = self._contrast(grepl(c[0], self.samples), grepl(c[1], self.samples), 'DE')

            # Make complex contrasts

            # Labels used for DE-TS and DE-AR
            de_diffs = grepl('-', contrasts.keys())

            # Now add DE-TS and DE-AR
            for de in de_diffs:
                # DE-TS
                base_de = list(map(lambda contrast: '(%s)' % contrast, contrasts[de]['contrasts']))
                contrasts["(%s)_ts" % de] = self._contrast(base_de[1:], base_de[:-1], fit_type='DE-TS')

                # DE-AR
                ar_de = [base_de[0]] * (len(base_de) - 1)
                contrasts["(%s)_ar" % de] = self._contrast(base_de[1:], ar_de, fit_type='DE-AR')
            expected_contrasts = contrasts
        else:
            expected_contrasts = list(map('-'.join, itertools.combinations(self.conditions, 2)))
        return expected_contrasts

    def suggest_contrasts(self):
        print('Timeseries Data:', self.timeseries)
        print('ts = Timeseries contrasts, ar = Autoregressive contrasts \n')
        if isinstance(self.default_contrasts, dict):
            sorted_keys = sorted(self.default_contrasts.keys())
            for k in sorted_keys:
                print(k +":", self.default_contrasts[k])
        elif isinstance(self.default_contrasts, list):
            for c in self.default_contrasts:
                print(c)

    @staticmethod
    def make_sample_ids(summary, reference_labels):
        """
        Make unique sample ID combinations.
        :param summary: dataframe; summary of experiments and samples. See get_experiment summary
        :param reference_labels
        :return:
        """
        ref_not_in_summary = [label for label in reference_labels if label not in summary.columns.values]
        if ref_not_in_summary:
            raise ValueError('Reference label(s) [%s] not found in the dataframe hierarchy variables.'
                             % ', '.join(ref_not_in_summary))
        # Make unique combinations from the reference labels
        combos = ['_'.join(combo) for combo in summary.loc[:, reference_labels].values.tolist()]
        combo_set = natsorted(list(set(combos)))
        ids = [combo_set.index(combo) for combo in combos]
        return ids, combos, combo_set

    @staticmethod
    def scale_to_baseline(df, zscore=False, **zkwargs):
        idx = list(df.columns.get_level_values('time')).index(0)
        scaled = df.divide(df.iloc[:, idx], axis=0).apply(np.log2)

        if zscore:
            # Can't zscore data if there is only one point in the vector
            if scaled.shape[1] > 1:
                normed = scaled.apply(zscore, axis=1, ddof=1, **zkwargs).fillna(value=0)
                scaled = normed

        return scaled

    def standardize(self):
        """
        Normalize the data to the 0 timepoint.

        NOTE: This should have expanded functionality in the future
        :return:
        """
        raw = self.data.copy()

        # Temporarily ignore invalid warnings. Zscore creates these if there are all uniform values (i.e. std=0)
        # It is corrected in the scale_to_baseline function
        np.seterr(invalid='ignore')
        standardized = self.data.groupby(axis=1, level=[0, 3]).apply(self.scale_to_baseline)
        np.seterr(invalid='warn')

        return standardized

    def print_experiment_summary(self, verbose=False):
        """
        Print a summary of the experimental details
        :param verbose: bool; whether or not to print the explicit values for each experimental variable.
                        Default (False) prints the number of unqiue values for each variable.
        :return:
        """
        for col in self.experiment_summary.columns:
            if verbose:
                try:
                    unique = sorted(np.array(list(set(self.experiment_summary[col]))).astype(int))
                except ValueError:
                    unique = sorted(list(set(self.experiment_summary[col])))
            else:
                unique = len(sorted(list(set(self.experiment_summary[col]))))
            print(col + "s:", unique)

    def _make_model_matrix(self, columns=None, formula='~0+x'):
        """
        Make the stats model matrix in R
        :param: formula: str; R formula character used to create the model matrix
        :return:    R-matrix
        """
        # Make an robject for the model matrix
        if columns is not None:
            r_sample_labels = robjects.FactorVector(columns)
            str_set = sorted(list(set(columns)))
        else:
            r_sample_labels = robjects.FactorVector(self.labels)
            str_set = sorted(list(set(self.labels)))

        # Create R formula object, and change the environment variable
        fmla = robjects.Formula(formula)
        fmla.environment['x'] = r_sample_labels

        # Make the design matrix. stats is a bound R package
        design = stats.model_matrix(fmla)
        design.colnames = robjects.StrVector(str_set)
        return design

    def _make_data_matrix(self, voom, log2=True):
        """
        Make the data matrix as an R object
        :return:
        """
        # Get the sample labels, genes, and data
        genes = self.raw.index.values

        if voom:
            r_data = rh.pydf_to_rmat(self.raw)
            voom_results = rh.unpack_r_listvector(limma.voom(r_data, save_plot=True))
            self.voom_results = voom_results
            data = voom_results['E']

            # Set the voom data
            self.data = voom_results['E'].copy()
            cols_as_tup = list(map(ast.literal_eval, self.data.columns.values))
            self.data.columns = pd.MultiIndex.from_tuples(cols_as_tup, names=self.raw.columns.names)
        else:
            # Log transform expression and correct values if needed
            if log2:
                data = np.log2(self.raw)
                # Save the new log2 data
            else:
                data = self.raw

            if np.sum(np.isnan(data.values)) > 0:
                warnings.warn("NaNs detected during log expression transformation. Setting NaN values to zero.")
                data = np.nan_to_num(data)
            if np.sum(np.isinf(data.values)) > 0:
                warnings.warn("infs detected during log expression transformation. Setting inf values to zero.")
                data.replace([np.inf, -np.inf], 0, inplace=True)

            # Set the data
            self.data = data

        r_matrix = rh.pydf_to_rmat(data)
        r_matrix.rownames = robjects.StrVector(genes)
        r_matrix.colnames = robjects.StrVector(self.labels)
        return r_matrix

    @staticmethod
    def _make_contrasts(contrasts, levels):
        """
        Make an R contrasts object that is used by limma

        :param contrasts: dict, list, str; The contrast(s) to use in differential expression.  A dictionary will be
            passed as kwargs, which is analagous to the ellipsis "..." in R.
        :return:
        """
        # If the contrasts are a dictionary they need to be unpacked as kwargs
        if isinstance(contrasts, dict):
            contrast_obj = limma.makeContrasts(**contrasts, levels=levels)
        # A string or list of strings can be passed directly
        else:
            contrast_obj = limma.makeContrasts(contrasts=contrasts, levels=levels)
        return contrast_obj

    @staticmethod
    def _ebayes(fit_obj, contrast_obj):
        """
        Calculate differential expression using empirical bayes
        :param fit_obj: MArrayLM; linear model fit_contrasts from limma in R. Typically from R function limma.lmFit()
        :param contrast_obj: R-matrix; numeric matrix with rows corresponding to coefficients in fit_contrasts and columns
            containing contrasts.
        :return:
        """
        contrast_fit = limma.contrasts_fit(fit=fit_obj, contrasts=contrast_obj)
        bayes_fit = limma.eBayes(contrast_fit)
        return bayes_fit

    def _fit_contrast(self, contrasts, samples):
        """
        Fit the differential expression model using the supplied contrasts.

        :param contrasts:  str, list, or dict; contrasts to test for differential expression. Strings and elements of
        lists must be in the format "X-Y". Dictionary elements must be in {contrast_name:"(X1-X0)-(Y1-Y0)").
        :return:
        """

        # Subset data and design to match contrast samples
        data = rh.pydf_to_rmat(rh.rvect_to_py(self.data_matrix).loc[:, samples])
        design = self._make_model_matrix(rh.rvect_to_py(data.colnames))
        # design = rh.rvect_to_py(self.design).loc[:, samples]
        # design = rh.pydf_to_rmat(design[(design == 1).any(axis=1)])

        # Setup contrast matrix
        contrast_robj = self._make_contrasts(contrasts=contrasts, levels=design)

        # Perform a linear fit_contrasts, then empirical bayes fit_contrasts
        linear_fit = limma.lmFit(data, design)
        fit = self._ebayes(linear_fit, contrast_robj)

        return fit

    def _fit_dict(self, contrast_dict) -> Dict[str, DEResults]:
        """
        Make the fits into a dictionary. This is wrapped into a function for type hinting
        :param self:
        :param names:
        :param contrasts:
        :return:
        """
        # Make fit dictionary
        fits = {name: DEResults(self._fit_contrast(contrast['contrasts'], contrast['samples']), name=name,
                                fit_type=contrast['fit_type']) for name, contrast in contrast_dict.items()}
        return fits

    def _samples_in_contrast(self, contrast: str, split_str='-') -> set:
        # Split the contrasts into individual samples
        s = contrast.split(split_str)

        # Make sure the samples match what is available in the data
        samples = set().union(*[grepl(ss.strip('(').strip(')'), self.samples) for ss in s])
        return samples

    def _make_fit_dict(self, contrasts, fit_names=None, force_separate=False) -> dict:
        """
        Make a dictionary of fits to conduct
        :param contrasts: 
        :param fit_names: list; Names for each fit. Only used when contrasts are a list. If none are supplied, 
        integers are used
        :param force_separate: bool; Only used when contrasts are a list. If True, force individual contrast items to
        be fit independently. 
        :return: 
        """

        # List
        if isinstance(contrasts, list):
            '''
            If it is a list, determine how many fits exist. If all items in the list aren't strings, then it is likely 
            multiple contrasts for independent fits
            '''
            n_fits = 1
            # Determine type of each item in list
            contrast_types = np.array([type(c) for c in contrasts])
            n_strings = sum(contrast_types == str)

            '''
            If not all of the list items are strings or user specifies to force separate, than each item will be a 
            separate fit
            '''
            if (n_strings != len(contrasts)) | force_separate:
                n_fits = len(contrasts)

            # Make a list of names
            if fit_names is None:
                if n_fits > 1:
                    names = [str(n) for n in range(n_fits)]
                else:
                    names = 0
            else:
                names = fit_names

            if n_fits == 1:
                fit_dict = {names: {'contrasts': contrasts,
                                    'samples': set().union(*[self._samples_in_contrast(c) for c in contrasts]),
                                    'fit_type': None}
                            }
            else:
                fit_dict = {name: {'contrasts': contrast,
                                   'samples': self._samples_in_contrast(contrast),
                                   'fit_type': None}
                            for name, contrast in zip(names, contrasts)
                            }

        # Str
        elif isinstance(contrasts, str):
            fit_dict = {contrasts: {'contrasts': contrasts,
                                    'samples': self._samples_in_contrast(contrasts),
                                    'fit_type': None}
                        }
        # Dict
        elif isinstance(contrasts, dict):
            fit_dict = contrasts

        else:
            raise TypeError('Contrasts must be supplied as str, list, or dict')

        return fit_dict

    def fit_contrasts(self, contrasts=None, fit_names=None, force_separate=False):
        """
        Wrapper to fit the differential expression model using the supplied contrasts.

        :param contrasts: str, list, or dict; contrasts to test for differential expression. Strings and elements of
        lists must be in the format "X-Y". Dictionary elements must be in {contrast_name:"(X1-X0)-(Y1-Y0)"). To conduct
        multiple fits, a list of mixed contrast types can be supplied. Each list item will be treated as an independent
        fit.
        :param names: str or list; names for each fit
        :return:
        """

        if self.data is None:
            raise ValueError('Please add data using set_data() before attempting to fit_contrasts.')

        # If there are no user supplied contrasts use the defaults
        if contrasts is None:
            contrasts = self.default_contrasts

        contrasts_to_fit = self._make_fit_dict(contrasts, fit_names=fit_names, force_separate=force_separate)

        # Add/Update Results dictionary
        self.results.update(self._fit_dict(contrasts_to_fit))
        self.contrast_dict = self.match_contrasts()
        self.db = self.make_results_db()

    def match_contrasts(self) -> dict:
        """
        Make a dictionary of {contrast: [DERs]} for easy referencing.

        :return: dict; format is {contrast: [DERs]}. Most values should be lists of length 1. Lists longer than 1
        indicate that the same contrast is present in multiple DERs. The values should be equivalent between DERs,
        however adjusted p-values may change slightly due to varied multiple hypothesis correction.
        """

        contrast_dict = {}
        for key, der in self.results.items():
            for c in der.contrast_list:
                if c in contrast_dict:
                    contrast_dict[c].append(key)
                else:
                    contrast_dict[c] = [key]
        return contrast_dict

    def make_results_db(self):
        """
        Compile the DER hierarchical results into a large dataframe, to rule them all. This should obviously be improved
        into a better datastructure
        :return:
        """
        der_results = [der.all_results for der in self.results.values()]
        db = pd.concat(der_results, axis=1, keys=self.results.keys(),
                       names=['fit', 'dtype', 'label'])                         # type: pd.DataFrame
        db.sort_index(axis=1, inplace=True, level=0)
        return db

    def to_pickle(self, path, force_save=False):
        # Note, this is taken directly from pandas generic.py which defines the method in class NDFrame
        """
        Pickle (serialize) object to input file path

        Parameters
        ----------
        path : string
            File path
        """
        should_pickle = True
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
