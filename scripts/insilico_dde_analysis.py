import sys

import pandas as pd
from pydiffexp import DEAnalysis

pd.set_option('display.width', 250)
dea = pd.read_pickle('intermediate_data/strongly_connected_2_dea.pkl')        # type: DEAnalysis
der = dea.results['ko-wt']
p_results = pd.read_pickle('intermediate_data/strongly_connected_2_ptest.pkl')

scores = p_results.loc[(der.top_table()['adj_pval'] < 0.05)]

# Remove clusters that have no dynamic DE (i.e. all 1, -1, 0)
interesting = scores.loc[scores.Cluster != '(0, 0, 0, 0, 0, 0, 0)']
print(interesting.sort_values(['Cluster', 'score'], ascending=False))
sys.exit()


# Heatmap of expression
de_data = (der.top_table().iloc[:, :7])#.multiply(der.p_value < 0.05)
sort_idx = interesting.sort_values(['Cluster', 'score'], ascending=False).index.values
# sort_idx = p_results.sort_values(['Cluster', 'score'], ascending=False).index.values
hm_data = de_data.loc[sort_idx]
hm_data = hm_data.divide(hm_data.abs().max(axis=1), axis=0)
# hm_data = stats.zscore(hm_data, ddof=1, axis=0)
print(hm_data.shape)

cmap = sns.diverging_palette(30, 260, s=80, l=55, as_cmap=True)
plt.figure(figsize=(4,8))
sns.heatmap(hm_data, xticklabels=dea.times, yticklabels=False, cmap=cmap)
plt.xticks(rotation=90)
plt.title('In silico DDE')
plt.tight_layout()
plt.show()
sys.exit()

hm_data = (der.top_table().iloc[:, :7])
# hm_data = hm_data.divide(hm_data.abs().max(axis=1), axis=0)
hm_data = stats.zscore(hm_data, ddof=1, axis=0)
sns.clustermap(hm_data, xticklabels=dea.times, yticklabels=False, cmap=cmap, col_cluster=False)
plt.xticks(rotation=90)
plt.title('In silico DDE')
plt.tight_layout()
plt.show()
