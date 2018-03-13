import os
import shutil
import tarfile

import numpy as np
import pandas as pd
from pydiffexp import DEAnalysis, DEResults
from pydiffexp.gnw import mk_ch_dir, GnwNetResults
from scipy import stats


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


def display():
    pass


if __name__ == '__main__':
    pd.set_option('display.width', 250)

    # Prep the raw data
    project_name = "GSE69822"
    prefix = "{}/{}_".format(project_name, project_name)
    contrast = 'ki-wt'
    raw = load_data('../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt')
    gene_map = pd.read_csv('../data/GSE69822/mcf10a_gene_names.csv', index_col=0)
    mk_ch_dir(project_name, ch=False)

    # Fit the data using a DEAnalysis object
    # dea = dde(raw, contrast, project_name, save_permute_data=True, calc_p=True, voom=True)

    # Compile simulation results
    # sim_stats = compile_sim('../data/motif_library/gnw_networks/', times=[0, 15, 40, 90, 180, 300],
    #                         save_path="{}{}_sim_stats.pkl".format(prefix, contrast))
    sim_stats = pd.read_pickle("{}{}_sim_stats.pkl".format(prefix, contrast))
    print(sim_stats)
    # dea = pd.read_pickle()

    # Display results
    display()