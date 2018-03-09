from os import listdir

import numpy as np
import pandas as pd
from pydiffexp import DEAnalysis, DEResults, get_scores
from pydiffexp.gnw.sim_explorer import tsv_to_dg
from scipy import stats


def get_data(path, c, n_timeseries, reps, perturbation_labels, t=None):
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
    df['rep'] = ts_rep_list

    idx = pd.IndexSlice
    full_data = df.set_index(['condition', 'rep', 'perturb', 'Time']).sort_index()

    if t is None:
        t = full_data.index.levels[full_data.index.names.index('Time')].values

    return full_data.loc[idx[:, :, :, t], :].copy()


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
        shuffled = grouped_data.apply(shuffle, axis=1)   # type: pd.DataFrame
        shuffled.to_pickle("{}/{}_permuted.pkl".format(save_dir, i))
    return


def run_fit(dea_obj, save_path):
    dea_obj.fit_contrasts()
    dea_obj.to_pickle(save_path)
    return


def analyze_permutes(real_scores, permutes_path) -> pd.DataFrame:
    grouped = real_scores.groupby('Cluster')
    p_score = pd.DataFrame()
    n_permutes = len(listdir(permutes_path))
    for p in listdir(permutes_path):
        # Load data and fit to get permuted data p-values
        p_idx = p.split("_")[0]
        print(p_idx)
        p_dea = DEAnalysis(pd.read_pickle("{}/{}".format(permutes_path, p)), time='Time', replicate='rep',
                           reference_labels=['condition', 'Time'], log2=False)
        p_dea.fit_contrasts()
        # Score ranking against the original clusters
        p_der = p_dea.results['ko-wt']
        weighted_lfc = (1 - p_der.p_value) * p_der.continuous.loc[p_der.p_value.index, p_der.p_value.columns]
        scores = get_scores(grouped, p_der.continuous.loc[:, p_der.p_value.columns], weighted_lfc).sort_index()
        scores['score'] = scores['score'] * (1 - p_der.continuous['adj_pval']).sort_index().values
        scores.sort_values('score', ascending=False, inplace=True)
        # drop cluster column, rename score, and add to real scores
        scores.drop('Cluster', inplace=True, axis=1)
        scores.columns = ['p{}_score'.format(p_idx)]
        p_score = pd.concat([p_score, scores], axis=1)

    p_mean = p_score.mean(axis=1)
    p_std = p_score.std(axis=1)
    values = real_scores['score'].copy()
    p_z = -1 * ((values - p_mean) / p_std).abs()
    p_values = (2 * p_z.apply(stats.norm.cdf)).round(decimals=int(n_permutes/10))
    p_values.name = 'p_value'
    new_scores = pd.concat([real_scores, p_values], axis=1)

    return new_scores


def shuffle(df, axis=0):
    df = df.copy()
    df.apply(np.random.shuffle, axis=axis, reduce=False)

    return df


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    t = [0, 15, 30, 60, 120, 240, 480]
    condition = ['wt', 'ko']
    base_dir = '../data/insilico/strongly_connected/'
    net_name = 'Yeast-100'
    ko_gene = 'YMR016C'
    reps = 3
    perturb = 'high_pos'

    # Set script run parameters
    n_p = 100           # Number of permutes
    p_path = "intermediate_data/strongly_connected_permutes"    # Path to save and retrieve permutes
    save_p = False       # Save permutes for future testing?
    fit_data = False     # Run analysis?
    analyze_p = True     # Analyze permuted data?

    # Organize perturbations
    perturbations = pd.read_csv("{}labeled_perturbations.csv".format(base_dir), index_col=0)
    p_labels = perturbations.index.values
    n_timeseries = len(p_labels)/reps

    df, dg = tsv_to_dg("{}Yeast-100.tsv".format(base_dir))

    data = pd.DataFrame()
    for c in condition:
        ts_file = '{bd}/{c}_sim/{nn}_{c}_dream4_timeseries.tsv'.format(bd=base_dir, nn=net_name, c=c)
        data = pd.concat([data, get_data(ts_file, c, n_timeseries, reps, p_labels)])
    idx = pd.IndexSlice

    # Censored object
    dea = DEAnalysis(data.sort_index().loc[idx[:, :, perturb, t], :].T, time='Time', replicate='rep',
                     reference_labels=['condition', 'Time'], log2=False)

    # Fit contrasts and save to pickle
    if fit_data:
        run_fit(dea, 'intermediate_data/strongly_connected_dea.pkl')

    # Save permuted data
    if save_p:
        grouped = dea.data.groupby(level='condition', axis=1)
        save_permutes("intermediate_data/strongly_connected_permutes/", grouped, n=n_p)

    if analyze_p:
        og_dea = pd.read_pickle('intermediate_data/strongly_connected_dea.pkl')         # type: DEAnalysis
        og_der = og_dea.results['ko-wt']                                                # type: DEResults
        og_scores = og_der.score_clustering()
        p_results = analyze_permutes(og_scores, "intermediate_data/strongly_connected_permutes/")
        p_results.to_pickle('intermediate_data/strongly_connected_ptest.pkl')