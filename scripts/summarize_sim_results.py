import ast
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from palettable.cartocolors.diverging import ArmyRose_7, Earth_7


def discretize_meta(df: pd.DataFrame):
    for col in df.columns:
        if 'logic' in col:
            val = np.zeros(len(df))
            val += df[col].str.contains('linear').values
            val -= df[col].str.contains('multiplicative').values
            df[col] = val
    return df.filter(regex='(->|logic)').astype(int)


if __name__ == '__main__':
    pd.set_option('display.width', 250)

    sim_info = pd.read_pickle('../data/motif_library/gnw_networks/simulation_info.pkl')         # type: pd.DataFrame
    motif_matching = pd.read_pickle('intermediate_data/strongly_connected_motif_match.pkl')     # type: pd.DataFrame
    motif_matching.set_index('id', inplace=True)
    sim_info.index = sim_info.index.astype(int)
    sim_info = discretize_meta(sim_info)
    sim_info = sim_info.loc[motif_matching.index.values]

    all_info = pd.concat([motif_matching, sim_info], axis=1)
    # all_info = all_info[all_info['mean']>0.5]
    all_info = all_info.groupby('true_gene')
    cmap = sns.diverging_palette(30, 260, s=80, l=55, as_cmap=True)

    dea = pd.read_pickle('intermediate_data/strongly_connected_dea.pkl')  # type: DEAnalysis
    der = dea.results['ko-wt']
    p_results = pd.read_pickle('intermediate_data/strongly_connected_ptest.pkl')
    scores = p_results.loc[(der.top_table()['adj_pval'] < 0.05) & (p_results['p_value'] < 0.05)]
    interesting = scores.loc[scores.Cluster.apply(ast.literal_eval).apply(set).apply(len) > 1]
    sort_idx = interesting.sort_values(['Cluster', 'score'], ascending=False).index.values

    # plt.figure(figsize=(4, 8))
    cmap1 = ArmyRose_7.mpl_colormap
    cmap2 = Earth_7.mpl_colormap
    # for g, data in all_info:
    #     print(data.shape)
    # sys.exit()
    edge_values = all_info.mean().iloc[:, [3, 5, 8, 9, 10, 11]].loc[sort_idx]
    logic_values = all_info.mean().iloc[:, [7, 4, 6]].loc[sort_idx]

    fig = plt.figure(figsize=(8, 6))
    parts = 3

    for i in range(parts):
        ax = plt.subplot2grid((1, parts*3), (0, 3*i), colspan=2)
        # ax = plt.subplot(gs[0, 2*i])
        eg = sns.heatmap(edge_values.iloc[:, [2*i, 2*i+1]], xticklabels=False, yticklabels=False,
                         cmap=cmap1, cbar=False, ax=ax)
        eg.set_ylabel('')
        ax = plt.subplot2grid((1, parts * 3), (0, i*3-1), colspan=1)
        log = sns.heatmap(pd.DataFrame(logic_values.iloc[:, i]), xticklabels=False, yticklabels=False,
                          cmap=cmap2, cbar=False, ax=ax)
        log.set_ylabel('')
    plt.subplots_adjust(wspace=0)
    plt.show()
    sys.exit()

