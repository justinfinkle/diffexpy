import os, sys, warnings, itertools
from typing import Dict
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from scipy.stats import zscore
from rpy2.robjects.packages import importr
from pydiffexp.utils.utils import int_or_float, grepl
import pydiffexp.utils.multiindex_helpers as mi
import pydiffexp.utils.rpy2_helpers as rh
from natsort import natsorted

# Activate conversion
rpy2.robjects.numpy2ri.activate()

# Load R packages
limma = importr('limma')
stats = importr('stats')

# Set null variable
null = robjects.r("NULL")


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
                warnings.warn('Cannot specify use_fstat=True when a specifiying a value for "coef"'
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


class DEAnalysis(object):
    """
    An object that does differential expression analysis with time course data
    """

    def __init__(self, df=None, index_names=None, split_str='_', time='time', condition='condition',
                 replicate='replicate', reference_labels=None):
        """

        :param df:
        :param index_names:
        :param split_str:
        :param time:
        :param condition:
        :param replicate:
        :param reference_labels:
        """

        self.data = None                    # type: pd.DataFrame
        self.labels = None
        self.times = None                   # type: list
        self.conditions = None              # type: list
        self.replicates = None              # type: list
        self.default_contrasts = None
        self.timeseries = False             # type: bool
        self.samples = None                 # type: list
        self.experiment_summary = None      # type: pd.DataFrame
        self.design = None                  # type: robjects.vectors.Matrix
        self.data_matrix = None             # type: robjects.vectors.Matrix
        self.contrast_robj = None           # type: robjects.vectors.Matrix
        self.fit = None                     # type: dict
        self.results = None                 # type: Dict[str, DEResults]
        self.decide = None                  # type: pd.DataFrame

        if df is not None:
            self._set_data(df, index_names=index_names, split_str=split_str, reference_labels=reference_labels)
            if time is not None:
                self.times = sorted(map(int_or_float, list(set(self.experiment_summary[time]))))
                if len(self.times) > 1:
                    self.timeseries = True
            self.conditions = sorted(list(set(self.experiment_summary[condition])))
            self.default_contrasts = self.possible_contrasts()
            self.replicates = sorted(list(set(self.experiment_summary[replicate])))

    def _set_data(self, df, index_names=None, split_str='_', reference_labels=None):
        # Check for a multiindex or try making one
        multiindex = mi.is_multiindex(df)
        h_df = None
        if sum(multiindex) == 0:
            h_df = mi.make_hierarchical(df, index_names=index_names, split_str=split_str)
        else:
            if multiindex[1]:
                h_df = df.copy()
            elif multiindex[0] and not multiindex[1]:  # Second part is probably redundant
                h_df = df.T
                warnings.warn('DataFrame transposed so multiindex is along columns.')

            # Double check multiindex
            multiindex = mi.is_multiindex(h_df)
            if sum(multiindex) == 0:
                raise ValueError('No valid multiindex was found, and once could not be created')

        h_df.sort_index(axis=1, inplace=True)
        self.data = h_df
        # Sort multiindex

        self.experiment_summary = self.get_experiment_summary(reference_labels=reference_labels)
        self.design = self._make_model_matrix()
        self.data_matrix = self._make_data_matrix()

    def get_experiment_summary(self, reference_labels=None):
        """
        Summarize the experiment details in a data frame
        :return:
        """
        index = self.data.columns
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
    def _contrast(v1, v2, fit_type, join_str='-'):
        contrast = {'contrasts': list(map(join_str.join, zip(v1, v2))),
                    'fit_type': fit_type}
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
            # TS-DE labels

            diffs = list(itertools.permutations(grepl(ts_str, contrasts.keys()), 2))
            # Add TS-AR labels
            diffs += list(itertools.permutations(grepl(ar_str, contrasts.keys()), 2))

            # Labels used for DE-TS and DE-AR
            de_diffs = grepl('-', contrasts.keys())

            for diff in diffs:
                ts1 = list(map(lambda contrast: '(%s)' % contrast, contrasts[diff[0]]['contrasts']))
                ts2 = list(map(lambda contrast: '(%s)' % contrast, contrasts[diff[1]]['contrasts']))
                fit_type = 'TS-DE' if ts_str in diff[0] else 'AR-DE'
                contrasts['-'.join(diff)] = self._contrast(ts1, ts2, fit_type=fit_type)

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
            expected_contrasts = list(map('-'.join, itertools.permutations(self.conditions, 2)))
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

    def standardize(self):
        """
        Normalize the data to the 0 timepoint.

        NOTE: This should have expanded functionality in the future
        :return:
        """
        raw = self.data.copy()
        for condition in self.conditions:
            # Standardize genes at each timepoint
            for tt in self.times:
                data = raw.loc(axis=1)[condition, :, tt, :]
                standard = np.nan_to_num(zscore(data, axis=0, ddof=1))
                raw.loc(axis=1)[condition, :, tt, :] = standard

            # Standardize genes across time
            for rep in self.replicates:
                data = raw.loc(axis=1)[condition, :, :, rep]

                if data.shape[1] > 2:
                    raw.loc(axis=1)[condition, :, :, rep] = zscore(data, axis=1, ddof=1)

        self.data = raw

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

    def _make_model_matrix(self, formula='~0+x'):
        """
        Make the stats model matrix in R
        :param: formula: str; R formula character used to create the model matrix
        :return:    R-matrix
        """
        # Make an robject for the model matrix
        r_sample_labels = robjects.FactorVector(self.labels)

        # Create R formula object, and change the environment variable
        fmla = robjects.Formula(formula)
        fmla.environment['x'] = r_sample_labels

        # Make the design matrix. stats is a bound R package
        design = stats.model_matrix(fmla)
        design.colnames = robjects.StrVector(sorted(list(set(self.labels))))
        return design

    def _make_data_matrix(self):
        """
        Make the data matrix as an R object
        :return:
        """
        # Get the sample labels, genes, and data
        genes = self.data.index.values

        # Log transform expression and correct values if needed
        data = np.log2(self.data.values)
        # data = self.data.values
        if np.sum(np.isnan(data)) > 0:
            warnings.warn("NaNs detected during log expression transformation. Setting to NaN values to zero.")
            data = np.nan_to_num(data)

        # Make r matrix object
        nr, nc = data.shape
        r_matrix = robjects.r.matrix(data, nrow=nr, ncol=nc)
        r_matrix.colnames = robjects.StrVector(self.labels)
        r_matrix.rownames = robjects.StrVector(genes)
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

    def _fit_contrast(self, contrast):
        """
        Fit the differential expression model using the supplied contrasts.

        :param contrast:  str, list, or dict; contrasts to test for differential expression. Strings and elements of
        lists must be in the format "X-Y". Dictionary elements must be in {contrast_name:"(X1-X0)-(Y1-Y0)").
        :return:
        """
        # Setup contrast matrix
        contrast_robj = self._make_contrasts(contrasts=contrast, levels=self.design)

        # Perform a linear fit_contrasts, then empirical bayes fit_contrasts
        linear_fit = limma.lmFit(self.data_matrix, self.design)
        fit = self._ebayes(linear_fit, contrast_robj)

        return fit

    def _fit_dict(self, contrast_dict: dict) -> Dict[str, DEResults]:
        """
        Make the fits into a dictionary. This is wrapped into a function for type hinting
        :param self:
        :param names:
        :param contrasts:
        :return:
        """

        # Make fit dictionary
        fits = {name: DEResults(self._fit_contrast(contrast['contrasts']), name=name, fit_type=contrast['fit_type'])
                for name, contrast in contrast_dict.items()}
        return fits

    def fit_contrasts(self, contrasts=None, names=None, fit_types=None, fit_defaults=True):
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

        # Initialize full contrast dictionary
        all_contrasts = self.default_contrasts if fit_defaults else {}

        if contrasts is not None:
            # Determine contrast type
            n_fits = 1
            if isinstance(contrasts, list):
                n_strings = sum([isinstance(l, str) for l in contrasts])

                # If all items in the list aren't strings, then it is likely multiple contrasts for independent fits
                if n_strings != len(contrasts):
                    n_fits = len(contrasts)

            # Make a list of names
            if names is None:
                names = [str(n) for n in range(n_fits)]
            elif isinstance(names, str):
                names = [names]

            # Make nested contrast dictionary to match format expected of default contrasts
            user_contrasts = {name: {'contrasts': contrast, 'fit_type': None} for name, contrast in zip(names, contrasts)}

            # Check if there are clashes and warn of override
            key_clash = [key for key in user_contrasts.keys() if key in all_contrasts.keys()]
            if key_clash:
                warning = ("\nThe user contrasts: '%s' are in the default contrasts "
                           "and will be overridden by the user values." % (','.join(key_clash)))
                warnings.warn(warning)
            all_contrasts = dict(all_contrasts, **user_contrasts)

        self.results = self._fit_dict(all_contrasts)

    def to_pickle(self, path):
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
        if os.path.isfile(path):
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
