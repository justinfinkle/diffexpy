__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import time
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pickle

class DEAnalysis(object):
    """
    An object that does differential expression analysis with time course data
    """
    def __init__(self, raw_data_path=None, sep='\t', skiprows=7, header=1, pre_filter_data=True, threshold=0.5,
                 background_correction=False, white_list=None):
        """
        Start the object by pointing at a csv with the following format:
            rows:       genes
            columns:    sample labels (eg. WT-Time, KO-Time)
        :param raw_data_path: str, optional
            The filepath for the delimited file that contains the data. Default is None. If None, data needs to be
            added manually
        :param sep: str, optional
            The delimitor used in the datafile. Refer to pandas documentation for allowed separators. Default is '\t'
            for tab-separated-values (tsv) to match Illumina GenomeStudio excel oputut.
        :param skiprows: int, optional
            Number of rows to skip in the file. Default is 7 to match with Illumina GenomeStudio excel output
        :param header: int, optional
            The row number that contains column headers. Default is 1 to match Illumina GenomeStudio excel output
        :return:
        """

        #Initialize features
        self.experiment_details = None
        self.condition_set = None
        self.replicate_set = None
        self.time_set = None
        self.n_conditions = None
        self.n_replicates = None
        self.n_times = None
        self.gene_names = None
        self.zscored_data = None

        if raw_data_path is not None: # Else you'll have to manually add the data
            # Load the data
            self.raw_data = self.load_data(raw_data_path, sep=sep, skiprows=skiprows, header=header)

            # Clean the data
            self.gene_data = self.clean_data(self.raw_data, background_correction=background_correction)

            # Make the data into an easily slicable dataframe. Can slice by replicate, condition, and/or time.
            self.full_data = self.make_data_dict(self.gene_data)

            #Zscore the data
            self.zscored_data = self.zscore_data(self.full_data)

            if pre_filter_data:
                self.filtered_zscore_data, self.filtered_genes, self.correlation_scores = self.filter_data(
                    self.zscored_data, threshold=threshold, white_list=white_list)

                self.filtered_data, _, _ = self.filter_data(self.full_data, False, self.correlation_scores,
                                                            threshold=threshold, white_list=white_list)

                self.average_zscore_data = self.make_average_data(self.filtered_zscore_data)
                self.average_data = self.make_average_data(self.filtered_data)
                for condition, genes in self.filtered_genes.items():
                    print("%i genes have a replicate correlation greater than %0.2f for %s"
                          %(len(genes), threshold, condition))

    def load_data(self, path, sep='\t', skiprows=7, header=1):
        """
        A simple wrapper to load the data. Perhaps unnecessary. See details for parameters in __init__
        :param path:
        :param sep:
        :param skiprows:
        :param header:
        :return:
        """
        df = pd.read_csv(path, sep=sep, skiprows=skiprows, header=header)
        return df

    def clean_data(self, data_df, gene_header='TargetID', background_correction=1):
        """
        Default method for cleaning the data. Built to work with Illumina GenomeStudio output
        :param data_df:
        :param gene_header:
        :return:
        """
        df = data_df.copy()
        gene_names = df[gene_header]

        # Find the columns that are the average signal and keep those
        clean_df = df[[col for col in df.columns if "AVG_Signal" in col]]
        clean_df.columns = self.replace_column_labels(clean_df.columns, 'Null', 'KO')
        clean_df.columns = self.replace_column_labels(clean_df.columns, 'AVG_Signal-BN-', '')

        column_df = self.get_experiment_summary(clean_df)

        # NOTE: This part is manual
        # Remove the two weird columns
        clean_df = clean_df.drop(['WT-1-1-0A', 'KO-16-1-0A'], 1)
        column_df = column_df.drop(['WT-1-1-0A', 'KO-16-1-0A'], 0)

        # Save the experiment details
        self.experiment_details = column_df

        # Do a background correction if requested
        if background_correction:
            clean_df[clean_df<background_correction] = background_correction

        # Put the gene names back in the dataframe
        clean_df.insert(0, 'TargetID', gene_names.copy())
        self.gene_names = gene_names.values

        return clean_df

    def get_experiment_summary(self, df):

        # Save the experiment details in the process
        column_df = pd.DataFrame([col.split('-') for col in df.columns], index=df.columns)

        # There is one extra timeopint for Replicate A. Correct the dataframe
        column_df.loc[:,2] = column_df.loc[:,2].replace('1', '0A')
        column_df.drop(3,1, inplace=True)
        column_df.columns = ['Condition', 'RunNum', 'Time_Replicate']

        return column_df

    def replace_column_labels(self, labels, to_replace, replace_with):
        """
        Simple function for generating list of new labels for a dataframe
        :param labels: 1-D array
            The exisiting labels
        :param to_replace: string
            The string that will be replaced in each label
        :param replace_with: string
            The string that will be inserted into each label
        :return:
        """
        new_labels = [label.replace(to_replace, replace_with) for label in labels]
        return new_labels

    def make_data_dict(self, df):
        """
        Return a dictionary
        :param df:
        :return:
        """
        df2 = self.make_sliceable_df(df)
        data_dict = {c: {(rep): df2[((df2['Replicate'] == rep) & (df2['Condition'] == c)) |
                    (df2['Replicate'] == 'Replicate')].T.iloc[:-3].set_index('Gene', drop=True)
                    for rep in self.replicate_set} for c in self.condition_set}

        return data_dict

    def make_sliceable_df(self, df):
        """
        Compile all the data into one sliceable dataframe. This makes for easy boolean slicing of the dataframe to
        select data that is of a certain replicate, condition, or time.
        :param df:
        :return:
        """
        column_df = self.experiment_details.copy()
        data_df = df.copy()

        times, replicates = zip(*[(int(col[:-1]), col[-1]) for col in column_df['Time_Replicate']])
        self.time_set = list(set(times))
        self.time_set.sort()
        self.replicate_set = list(set(replicates))
        self.condition_set = list(set(column_df['Condition']))
        self.n_replicates = len(self.replicate_set)
        self.n_times = len(self.time_set)
        self.n_conditions = len(self.condition_set)

        column_df['Times'] = times
        column_df['Replicate'] = replicates

        # Slice the data_df columns in the order we want (every third)
        n_columns = len(data_df.columns)
        idx = list(range(1, n_columns, 3))+list(range(2, n_columns, 3))+list(range(3, n_columns, 3))

        df = data_df.iloc[:, idx]
        df.insert(0, 'Gene', data_df['TargetID'])
        df = df.append([column_df['Times'], column_df['Replicate'], column_df['Condition']])
        df.loc['Times', 'Gene'] = 'Time'
        df.loc['Replicate', 'Gene'] = 'Replicate'
        df.loc['Condition', 'Gene'] = 'Condition'

        return df.T

    def replicate_correlation(self, condition, data_dict):
        pair_combos = itertools.combinations(self.replicate_set, 2)
        pearson_df = None
        print('Calculating pearson correlation for replicates in %s...'%condition)
        for combo in pair_combos:
            p = self.pearson_correlation(data_dict[combo[0]].values, data_dict[combo[1]].values)
            if pearson_df is None:
                pearson_df = pd.DataFrame(p, columns=[combo])
            else:
                pearson_df[combo] = p
        pearson_df['Median_pearson'] = np.median(pearson_df, axis=1)
        pearson_df['Mean_pearson'] = np.mean(pearson_df, axis=1)
        pearson_df.index = self.gene_names
        print('[DONE]')
        return pearson_df

    def keep_genes(self, pearson_df, metric='Mean_pearson', threshold=0.5):
        keep_idx = pearson_df.index[pearson_df[metric]>=threshold]
        return keep_idx

    def pearson_correlation(self, data1, data2):
        num_genes = len(data1)
        pearson_coef = np.array([stats.pearsonr(data1[ii], data2[ii])[0] for ii in range(num_genes)])
        return pearson_coef

    def zscore_data(self, data_dict):
        zscore_df = {k: {key: pd.DataFrame(stats.zscore(value.values.astype(float), axis=1, ddof=1),
                                           columns=value.columns, index=value.index)
                         for key, value in v.items()} for k, v in data_dict.items()}
        return zscore_df

    def stack_data_replicates(self, data_dict):
        """
        This function is meant to return a dictionary that stacks replicates. There should be a data array for each
        condition
        :param data_dict:
        :return:
        """
        stacked_data = {}   # Initialize
        for k, v in data_dict.items():
            stacked_data[k] = {'values':None, 'genes':None, 'features':None, 'replicate_order':[]}    # Initialize
            for key, value in v.items():
                if key in self.replicate_set:
                    stacked_data[k]['replicate_order'].append(key)
                    if stacked_data[k]['values'] is None:
                        stacked_data[k]['values'] = value.values
                        stacked_data[k]['genes'] = value.index
                        stacked_data[k]['features'] = value.columns
                    else:
                        stacked_data[k]['values'] = np.dstack((stacked_data[k]['values'], value.values))
        return stacked_data

    def ttest_noise(self, data, null_mean=0, p_threshold=0.05):
        # Calculate one tailed p values for each data point based on replicates
        t_scores, p_vals = stats.ttest_1samp(data.astype(float), null_mean, axis=2)
        one_tailed_p = p_vals/2
        zeroed_data = data.copy()
        zeroed_data[one_tailed_p>=p_threshold] = null_mean
        return zeroed_data

    def log_fold_change(self, data):
        fold_change = np.zeros(data.shape)

        # Log fold change from the zero time point
        for rep in range(data.shape[2]):
            current_slice = data[:, :, rep]
            first_vector = current_slice[:, 0]
            fold_change[:, :, rep] = current_slice.astype(np.float64)/(first_vector[:, None]).astype(np.float64)

        return np.log2(fold_change)

    def log_fold_change_data(self, stacked_data, remove_noise_value=0):
        log_data = stacked_data.copy()
        for key, value in log_data.items():
            if remove_noise_value:
                noise_filtered_data = self.ttest_noise(value['values'], remove_noise_value)
            else:
                noise_filtered_data = value['values']
            log_data[key]['values'] = self.log_fold_change(noise_filtered_data)
        return log_data

    def stacked_to_data_dict(self, stacked_data, replace_nan_inf=True, replace_rep=False):
        data_dict = {}
        for key, value in stacked_data.items():
            data_dict[key]={}
            for ii, rep in enumerate(value['replicate_order']):
                if replace_rep:
                    current_features = [feature.replace('A', rep) for feature in value['features']]
                else:
                    current_features = value['features']
                current_data = value['values'][:, :, ii]
                if replace_nan_inf:
                    current_data[np.isinf(current_data)] = 0
                    current_data = np.nan_to_num(current_data)
                data_dict[key][value['replicate_order'][ii]] = pd.DataFrame(current_data, index=value['genes'],
                                                                             columns=current_features)
        return data_dict

    def filter_data(self, data_dict, calc_correlation_scores=True, correlation_dict=None, threshold=0.5,
                    metric='Mean_pearson', white_list=False):
        #todo: There is probably a more elegant way to do this
        """
        Filter the data dictionary
        :param data_dict:
        :param calc_correlation_scores:
        :param correlation_dict:
        :param threshold:
        :param metric:
        :return:
        """
        if white_list:
            white_list = set(white_list)
        if calc_correlation_scores:
            correlation_dict = self.make_correlation_scores(data_dict)
        if correlation_dict is None:
            raise ValueError('If calc_correlation_scores if False a correlation dictionary must be provided')

        filtered_data = {}
        filtered_genes = {}
        for condition, replicate_dict in data_dict.items():
            keep_genes = set(self.keep_genes(correlation_dict[condition], metric=metric, threshold=threshold))
            if white_list:
                keep_genes = keep_genes.union(white_list)
            filtered_genes[condition] = keep_genes
            filtered_data[condition] = {}
            for key, value in replicate_dict.items():
                filtered_data[condition][key] = value.loc[keep_genes]
        intersection = None
        union = None
        for condition, kept_genes in filtered_genes.items():
            if intersection is None:
                intersection = kept_genes.copy()
            else:
                intersection = intersection.intersection(kept_genes)
            if union is None:
                union = kept_genes.copy()
            else:
                union = union.union(kept_genes)
        if white_list:
            intersection = intersection.union(white_list)
            union = union.union(white_list)
        filtered_genes['intersection'] = intersection
        filtered_genes['union'] = union
        for condition, replicate_dict in data_dict.items():
            filtered_data['intersection_'+condition] = {}
            filtered_data['union_'+condition] = {}
            for key, value in replicate_dict.items():
                filtered_data['intersection_'+condition][key] = value.loc[intersection]
                filtered_data['union_'+condition][key] = value.loc[union]

        return filtered_data, filtered_genes, correlation_dict

    def make_correlation_scores(self, data_dict):
        """
        Calculate correlation scores
        :param data_dict:
        :return:
        """
        correlation_dict = {}
        for key, value in data_dict.items():
            correlation_dict[key] = self.replicate_correlation(key, value)

        return correlation_dict

    def make_average_data(self, data_dict):
        """
        Calculate the average of replicates
        :param data_dict:
        :return:
        """
        stacked_data = self.stack_data_replicates(data_dict)
        average_data = {}
        for key, value in stacked_data.items():
            average_data[key] = pd.DataFrame(np.mean(value['values'], axis=2),
                                             index=value['genes'], columns=value['features'])

        return average_data

    def to_pickle(self, path):
        # Note, this is taken direclty from pandas generic.py which defines the method in class NDFrame
        """
        Pickle (serialize) object to input file path
        Parameters
        ----------
        path : string
            File path
        """
        from pandas.io.pickle import to_pickle
        return to_pickle(self, path)