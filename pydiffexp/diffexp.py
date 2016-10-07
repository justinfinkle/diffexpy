import os, sys, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
rpy2.robjects.numpy2ri.activate()


def is_multiindex(df):
    """
    Function to determine if a dataframe is multiindex
    :param df: dataframe
    :return: tuple
    """
    mi = [False, False]
    mi[0] = True if isinstance(df.index, pd.MultiIndex) else False
    mi[1] = True if isinstance(df.columns, pd.MultiIndex) else False
    return tuple(mi)


def make_hierarchical(df, index_names=None, split_str='_'):
    """

    Parameters
    ----------
    df
    index_names
    split_str

    Returns
    -------

    """
    """
    Make a regular dataframe hierarchical by adding a MultiIndex
    :param df: dataframe; the dataframe to made hierarchical
    :param index_names: list; names for each of the categories of the multiindex
    :param axis: int (0 or 1); axis along which to split the index into a multiindex. Default (0) splits along the dataframe index, while 1 splits along the dataframe columns
    :param split_str: str; the string on which to split tuples
    :return: dataframe; hierarchical dataframe with multiindex
    """

    # Split each label into hierarchy
    try:
        index = df.columns
        s_index = split_index(index, split_str)
    except ValueError:
        df = df.T
        index = df.columns
        s_index = split_index(index, split_str)
        warnings.warn('Multiindex found for rows, but not columns. Returned data frame is transposed from input')

    h_df = df.copy()
    m_index = pd.MultiIndex.from_tuples(s_index, names=index_names)
    h_df.columns = m_index

    return h_df


def split_index(index, split_str):
    """
    Split a list of strings into a list of tuples.
    :param index: list-like; List of strings to be split
    :param split_str: str; substring by which to split each string
    :return:
    """
    s_index = [tuple(ind.split(split_str)) for ind in index if split_str in ind]
    if len(s_index) != len(index):
        raise ValueError('Index not split properly using supplied string')
    return s_index


class DEAnalysis(object):
    """
    An object that does differential expression analysis with time course data
    """

    def __init__(self, df=None, index_names=None, split_str='_', reference_labels=None):
        self.data = None
        self.sample_labels = None
        self.contrasts = None
        self.experiment_summary = None
        self.design = None
        self.data_matrix = None
        self.contrast_robj = None
        self.l_fit = None
        self.de_fit = None
        self.results = None

        if df is not None:
            self._set_data(df, index_names=index_names, split_str=split_str, reference_labels=reference_labels)

        # Import requisite R packages
        self.limma = importr('limma')
        self.stats = importr('stats')

    def _set_data(self, df, index_names=None, split_str='_', reference_labels=None):
        # Check for a multiindex or try making one
        multiindex = is_multiindex(df)
        h_df = None
        if sum(multiindex) == 0:
            h_df = make_hierarchical(df, index_names=index_names, split_str=split_str)
        else:
            if multiindex[1]:
                h_df = df.copy()
            elif multiindex[0] and not multiindex[1]:  # Second part is probably redundant
                h_df = df.T
                warnings.warn('DataFrame transposed so multiindex is along columns.')

            # Double check multiindex
            multiindex = is_multiindex(h_df)
            if sum(multiindex) == 0:
                raise ValueError('No valid multiindex was found, and once could not be created')

        self.data = h_df
        self.experiment_summary = self.get_experiment_summary(reference_labels=reference_labels)

    def get_experiment_summary(self, reference_labels=None):
        """
        Summarize the experiment details in a data frame
        :return:
        """
        index = self.data.columns
        summary_df = pd.DataFrame()
        for ii, name in enumerate(index.names):
            summary_df[name] = index.levels[ii].values[index.labels[ii]]
        if reference_labels is not None:
            summary_df['sample_id'], self.sample_labels = self.make_sample_ids(summary_df,
                                                                               reference_labels=reference_labels)
        else:
            warnings.warn('Sample IDs and labels not set because no reference labels supplied. R data matrix and '
                          'contrasts cannot be created without sample IDs. Setting sample labels to integers')
            self.sample_labels = list(map(lambda x: 'x%i' % x, range(len(summary_df))))
        return summary_df

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
        combo_set = sorted(list(set(combos)))
        ids = [combo_set.index(combo) for combo in combos]
        return ids, combos

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
        r_sample_labels = robjects.FactorVector(self.sample_labels)

        # Create R formula object, and change the environment variable
        fmla = robjects.Formula(formula)
        fmla.environment['x'] = r_sample_labels

        # Make the design matrix. self.stats is a bound R package
        design = self.stats.model_matrix(fmla)
        design.colnames = robjects.StrVector(sorted(list(set(self.sample_labels))))
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
        if np.sum(np.isnan(data)) > 0:
            warnings.warn("NaNs detected during log expression transformation. Setting to NaN values to zero.")
            data = np.nan_to_num(data)

        # Make r matrix object
        nr, nc = data.shape
        r_matrix = robjects.r.matrix(data, nrow=nr, ncol=nc)
        r_matrix.colnames = robjects.StrVector(self.sample_labels)
        r_matrix.rownames = robjects.StrVector(genes)
        return r_matrix

    def _make_contrasts(self, contrasts, levels):
        """
        Make an R contrasts object that is used by limma
        :param contrasts: dict, list, str; The contrast(s) to use in differential expression.  A dictionary will be
            passed as kwargs, which is analagous to the ellipsis "..." in R.
        :return:
        """
        # If the contrasts are a dictionary they need to be unpacked as kwargs
        if isinstance(contrasts, dict):
            contrast_obj = self.limma.makeContrasts(**contrasts,
                                                    levels=levels)
        # A string or list of strings can be passed directly
        else:
            contrast_obj = self.limma.makeContrasts(contrasts=contrasts,
                                                    levels=levels)
        return contrast_obj

    def _ebayes(self, fit_obj, contrast_obj):
        """
        Calculate differential expression using empirical bayes
        :param fit_obj: MArrayLM; linear model fit from limma in R. Typically from R function limma.lmFit()
        :param contrast_obj: R-matrix; numeric matrix with rows corresponding to coefficients in fit and columns
            containing contrasts.
        :return:
        """
        contrast_fit = self.limma.contrasts_fit(fit=fit_obj, contrasts=contrast_obj)
        bayes_fit = self.limma.eBayes(contrast_fit)
        return bayes_fit

    def get_results(self, use_fstat=None, p_value=0.05, n='inf', **kwargs):
        """
        Print get_results of differential expression analysis
        :param use_fstat: bool; select genes using F-statistic. Useful if testing significance for multiple contrasts,
        such as a time series
        :param p_value float; cutoff for significant get_results. Default is 0.05. If np.inf, then no cutoff is applied
        :param n int or 'inf'; number of significant get_results to include in output. Default is 'inf' which includes all
        get_results passing the threshold
        :param kwargs: additional arguments to pass to topTable. see topTable documentation in R for more details.
        :return:
        """

        # Update kwargs with commonly used ones provided in this API
        kwargs = dict(kwargs, p_value=p_value, n=n)

        # Use fstat if multiple contrasts supplied
        if use_fstat is None:
            use_fstat = False if (isinstance(self.contrasts, str) or
                                  (isinstance(self.contrasts, list) and len(self.contrasts) > 1)) else True

        if use_fstat:
            if 'coef' in kwargs.keys():
                raise ValueError('Cannot specify value for argument "coef" when using F statistic for topTableF')
            table = self.limma.topTableF(self.de_fit, **kwargs)
        else:
            table = self.limma.topTable(self.de_fit, **kwargs)

        with localconverter(default_converter + pandas2ri.converter) as cv:
            df = pandas2ri.ri2py(table)
        return df

    def fit(self, contrasts):
        """
        Fit the differential expression model using the supplied contrasts
        :param contrasts: str, list, or dict; contrasts to test for differential expression. Strings and elements of
        lists must be in the format "X-Y". Dictionary elements must be in {contrast_name:"(X1-X0)-(Y1-Y0)")
        :return:
        """
        # Save the user supplied contrasts
        self.contrasts = contrasts
        if self.data is None:
            raise ValueError('Please add data using set_data() before attempting to fit.')

        # Setup design, data, and contrast matrices
        self.design = self._make_model_matrix()
        self.data_matrix = self._make_data_matrix()
        self.contrast_robj = self._make_contrasts(contrasts=self.contrasts, levels=self.design)

        # Perform a linear fit, then empirical bayes fit
        self.l_fit = self.limma.lmFit(self.data_matrix, self.design)
        self.de_fit = self._ebayes(self.l_fit, self.contrast_robj)

        # Set results to include all test values
        self.results = self.get_results(p_value=np.inf)

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
