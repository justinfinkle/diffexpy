import ast

import numpy as np
import pandas as pd
from pydiffexp import get_scores, DEResults
from pydiffexp.analyze import pairwise_corr
from scipy import stats

calc_correlation = False
sim_data = pd.read_csv('intermediate_data/sim_stats_censoredtimes.tsv', sep='\t', index_col=[0, 1, 2], header=[0,1])
dea = pd.read_pickle('intermediate_data/strongly_connected_dea.pkl')
der = dea.results['ko-wt']              # type: DEResults

sim_data.fillna(0, inplace=True)

# Remove clusters that have no dynamic DE (i.e. all 1, -1, 0)
p_results = pd.read_pickle('intermediate_data/strongly_connected_ptest.pkl')
scores = p_results.loc[(der.top_table()['adj_pval'] < 0.05) & (p_results['p_value'] < 0.05)]
interesting = scores.loc[scores.Cluster.apply(ast.literal_eval).apply(set).apply(len) > 1]
filtered_data = dea.data.loc[interesting.index]

if calc_correlation:
    print('Computing pairwise')
    gene_mean = filtered_data.groupby(level=['condition', 'Time'], axis=1).mean()
    gene_mean_grouped = gene_mean.groupby(level='condition', axis=1)
    mean_z = gene_mean_grouped.transform(stats.zscore, ddof=1).fillna(0)

    # Correlate zscored means for each gene with each node in every simulation
    sim_means = sim_data.loc[:, ['ko_mean', 'wt_mean']]
    sim_mean_z = sim_means.groupby(level='stat', axis=1).transform(stats.zscore, ddof=1).fillna(0)

    pcorr = pairwise_corr(sim_mean_z, mean_z, axis=1).T
    pcorr.to_pickle('intermediate_data/strongly_connected_sim_corr.pkl')
else:
    pcorr = pd.read_pickle('intermediate_data/strongly_connected_sim_corr.pkl')

# Cluster and rank the simulations
weighted_lfc = ((1 - sim_data.loc[:, 'lfc_pvalue']) * sim_data.loc[:, 'lfc'])
sim_discrete = sim_data.loc[:, 'lfc'].apply(np.sign).fillna(0).astype(int)
sim_clusters = der.cluster_discrete((sim_discrete*(sim_data.loc[:, 'lfc_pvalue']<0.05)))
sim_g = sim_clusters.groupby('Cluster')
sim_scores = get_scores(sim_g, sim_data.loc[:, 'lfc'], weighted_lfc).sort_index()
sim_interesting = sim_scores.loc[sim_scores.Cluster.apply(ast.literal_eval).apply(set).apply(len) > 1]

# print(sim_interesting.sort_values(['Cluster', 'score'], ascending=False))

sim_interesting.set_index(['x_perturbation', 'id'], append=True, inplace=True)
sim_interesting = sim_interesting.swaplevel(i='id', j='gene')
sim_interesting.sort_index(inplace=True)
pcorr.sort_index(inplace=True)

pd.set_option('display.width', 250)
idx = pd.IndexSlice
matching_results = pd.DataFrame()
for gene, row in interesting.iterrows():
    candidate_nets = sim_interesting.loc[sim_interesting.Cluster == row.Cluster]

    ranking = pd.concat([candidate_nets, pcorr.loc[candidate_nets.index, gene]], axis=1)
    ranking['mean'] = (ranking['score'] + ranking[gene])/2
    ranking = ranking.loc[ranking.index.get_level_values(2) == 'y']
    ranking['true_gene'] = gene

    matching_results = pd.concat([matching_results, ranking.reset_index()], ignore_index=True, join='inner')
# Save matching results
matching_results.to_pickle('intermediate_data/strongly_connected_motif_match.pkl')