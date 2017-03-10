import sys
import pandas as pd
from collections import Counter
import numpy as np
from pydiffexp.utils.io import read_dea_pickle
from pydiffexp.utils.utils import column_unique
import matplotlib.pyplot as plt
from pydiffexp import DEPlot

pd.set_option('display.width', 1000)

# Load the DEAnalysis object with fits and data
dea = read_dea_pickle("./sprouty_pickle.pkl")
print(dea.results['KO-WT'].F.shape)
sys.exit()
# dep = DiffExpPlot(dea)#
# idx = pd.IndexSlice
#
# # print(dea.results['KO-WT'].top_table(p=0.05).shape)
# # dep.volcano_plot(dea.results['KO-WT'].top_table(coef=1), top_n=15)
# # plt.tight_layout()
# # plt.show()
# # sys.exit()
#
# # Subselect using the discrete values
# # Prioritize by continuous values
# x = dea.db.loc[:, idx[:, :, 'Cluster']]
# clusters = [tuple(gene) for gene in x.values]
# c_idx = pd.MultiIndex.from_tuples(clusters, names=x.columns.levels[0])
# g = pd.DataFrame(x.index.values, columns=['Gene'])
# g.index = c_idx
# grouped = g.groupby(level=['(KO-WT)_ar', 'KO-WT', '(KO-WT)_ts'])
# g.sort_index(level=0, inplace=True)
# z = g.loc[idx['(0, 0, 0, 0)', :, '(1, 1, 1, 1, 1)'], 'Gene']
# enrich = dea.db.loc[z, idx['KO-WT', 'continuous']].sort_values('adj_pval')
# print(enrich)
# enrich = enrich[enrich['adj_pval']<0.001]
# # print(enrich)
# # for ii in enrich.index.values:
# #     print(ii)
#
# gene = 'TEAD2'
# data = dea.data.loc[gene]
# print(dea.db.loc[gene, idx[:, :, 'Cluster']])
# dep.tsplot(data)
# plt.tight_layout()
# plt.show()
#
# sys.exit()

# ======================================================================================================================
# ======================================================================================================================
# Genes that are initially DE
# ======================================================================================================================
# ======================================================================================================================

# Useful params
p_val = 0.001
ic = 'KO_0-WT_0'    # Initial contrast
der = dea.results['KO-WT']

# This represents traditional analysis because it only considers the first time point
de_ko = der.top_table(coef=1, p=p_val, use_fstat=False)
print("%i genes DE initially\n" % len(de_ko), de_ko.head())


# If we consider additional timepoints the results change slightly because of the global multiple hypothesis correction
de_ko_global = der.discrete.loc[:, ic]
de = de_ko_global==0
print('\n', len(de_ko_global[de]), 'DE when considering global effects')

# Genes that start off differentially expressed but no longer are
converge = der.discrete[(de) & (der.discrete==0).any(axis=1)]
genes = der.cluster_discrete(converge)[der.cluster_discrete(converge)['Cluster'] == '(0, 0, 1, 1,  1)']
print(der.continuous.loc[genes.index].sort_values('adj_pval'))

# der = dea.results['(KO-WT)_ar']
# discrete = der.decide_tests()
# genes = der.cluster_discrete(discrete)[(der.cluster_discrete(discrete)['Cluster'] == '(1, 1, 0, 0)') & de]
# print(der.continuous.loc[genes.index].sort_values('adj_pval'))

dep = DiffExpPlot(dea)

x = dea.data.loc['CISH']
dep.tsplot(x)
plt.tight_layout()
plt.show()




