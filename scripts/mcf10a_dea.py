from os import listdir

import numpy as np
import pandas as pd
from pydiffexp import DEAnalysis, get_scores
from scipy import stats


def clean_data():
    replace_dict = ["A66 no EGF", "PIK3CA H1047R", "PTEN KO"]
    out = open('../data/GSE69822/GSE69822_RNA-Seq_RPKMs_cleaned.txt', 'w')
    with open('../data/GSE69822/GSE69822_RNA-Seq_RPKMs.txt') as file:
        data = file.readlines()
        for line in data:
            newline = line
            for r in replace_dict:
                newline = newline.replace(r, r.replace(" ", "_"))
            out.write(newline)
    out.close()


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


def analyze_permutes(real_scores, permutes_path) -> pd.DataFrame:
    grouped = real_scores.groupby('Cluster')
    p_score = pd.DataFrame()
    n_permutes = len(listdir(permutes_path))
    for p in listdir(permutes_path):
        # Load data and fit to get permuted data p-values
        p_idx = p.split("_")[0]
        print(p_idx)
        p_dea = DEAnalysis(raw_data, index_names=hierarchy, reference_labels=['condition', 'time'], log2=False)
        p_dea.fit_contrasts(p_dea.default_contrasts['pten-wt']['contrasts'], fit_names='pten-wt')
        # Score ranking against the original clusters
        p_der = p_dea.results['pten-wt']
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
    new_scores = pd.concat([real_scores, p_values], axis=1)         # type: pd.DataFrame

    return new_scores


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    # Set script run parameters
    n_p = 100           # Number of permutes
    p_path = "intermediate_data/GSE69822_permutes/"    # Path to save and retrieve permutes
    save_p = False       # Save permutes for future testing?
    fit_data = True     # Run analysis?
    analyze_p = True     # Analyze permuted data?

    # Load original data
    raw_data = pd.read_csv('../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt', sep=',', index_col=0)
    hierarchy = ['condition', 'replicate', 'time']

    # The example data has been background corrected, so set everything below 0 to a trivial positive value of 1
    raw_data[raw_data <= 0] = 1
    dea = DEAnalysis(raw_data, index_names=hierarchy, reference_labels=['condition', 'time'], voom=True)

    # Fit contrasts and save to pickle
    if fit_data:
        dea.fit_contrasts(dea.default_contrasts['pten-wt']['contrasts'], fit_names='pten-wt')
        dea.to_pickle('intermediate_data/GSE69822_dea.pkl')

    # Save permuted data
    if save_p:
        idx = pd.IndexSlice
        data = dea.data.loc[:, idx[['wt', 'pten'], :, :]]
        grouped = data.groupby(level='condition', axis=1)
        save_permutes(p_path, grouped, n=n_p)

    if analyze_p:
        og_dea = pd.read_pickle('intermediate_data/GSE69822_dea.pkl')         # type: DEAnalysis
        og_der = og_dea.results['pten-wt']                                                # type: DEResults
        og_scores = og_der.score_clustering()
        p_results = analyze_permutes(og_scores, "intermediate_data/GSE69822_permutes/")
        p_results.to_pickle('intermediate_data/GSE69822_ptest.pkl')