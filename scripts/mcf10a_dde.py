import ast
import sys

import matplotlib.pyplot as plt
import pandas as pd
from pydiffexp import DEAnalysis, DEPlot

pd.set_option('display.width', 250, 'display.max_rows', 500)
dea = pd.read_pickle('intermediate_data/GSE69822_dea.pkl')        # type: DEAnalysis
der = dea.results['pten-wt']
p_results = pd.read_pickle('intermediate_data/GSE69822_ptest.pkl')
gene_names = pd.read_csv('../data/GSE69822/GSE69822_RNA-Seq_RPKMs_cleaned.txt', sep=' ')
gene_names = gene_names[['id', 'hgnc_symbol']].set_index('id').drop_duplicates()
p_thresh = 0.001
scores = p_results.loc[(der.top_table()['adj_pval'] < p_thresh) & (p_results['p_value'] < p_thresh)]

# Remove clusters that have no dynamic DE (i.e. all 1, -1, 0)
interesting = scores.loc[scores.Cluster.apply(ast.literal_eval).apply(set).apply(len) > 1]
# print(interesting.sort_values(['score'], ascending=False).head(100))
c = (interesting[interesting.Cluster=='(0, 0, 0, 0, -1, -1)'].sort_values('score', ascending=False))
print(c)
dep = DEPlot()
dep.tsplot(dea.data.loc["ENSG00000186187", ['pten', 'wt']], legend=False)
plt.title('ZNRF1')
plt.tight_layout()
plt.show()
sys.exit()
for gene in c.index:
    dep.tsplot(dea.data.loc[gene, ['pten', 'wt']], legend=False)
    plt.tight_layout()
    plt.show()
sys.exit()

# Heatmap of expression
de_data = (der.top_table().iloc[:, :6])#.multiply(der.p_value < p_thresh)
sort_idx = interesting.sort_values(['Cluster', 'score'], ascending=False).index.values
hm_data = de_data.loc[sort_idx]
hm_data = hm_data.divide(hm_data.abs().max(axis=1), axis=0)

cmap = sns.diverging_palette(30, 260, s=80, l=55, as_cmap=True)
plt.figure(figsize=(4, 8))
sns.heatmap(hm_data, xticklabels=dea.times, yticklabels=False, cmap=cmap)
plt.xticks(rotation=45)
plt.title('PTEN KO DDE')
plt.tight_layout()
plt.show()
sys.exit()

for g, data in interesting.groupby('Cluster'):
    print(g, data.shape)
    print(data.sort_values('score', ascending=False))
    for gene in data.index:
        try:
            print(gene_names.loc[gene, 'hgnc_symbol'])
        except:
            print(gene)
