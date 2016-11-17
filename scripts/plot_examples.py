import sys, warnings
import pandas as pd
import numpy as np
from pydiffexp import DEAnalysis, volcano_plot, tsplot
import discretized_clustering as dcluster
import matplotlib.pyplot as plt


def grepl(search_list, substr):
    substr = substr.lower()
    grep_list = list(filter(lambda x: substr in x.lower(), search_list))
    return grep_list

# Variables
test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/data/raw_data/all_data_formatted.csv"
raw_data = pd.read_csv(test_path, index_col=0)
hierarchy = ['condition', 'well', 'time', 'replicate']

raw_data[raw_data <= 0] = 1
dea = DEAnalysis(raw_data, index_names=hierarchy, reference_labels=['condition', 'time'])

# Types of contrasts
c_dict = {'Diff0': "(KO_15-KO_0)-(WT_15-WT_0)", 'Diff15': "(KO_60-KO_15)-(WT_60-WT_15)",
          'Diff60': "(KO_120-KO_60)-(WT_120-WT_60)", 'Diff120': "(KO_240-KO_120)-(WT_240-WT_120)"}
# c_list = ["KO_0-WT_0", "KO_15-WT_15", "KO_60-WT_60", "KO_120-WT_120", "KO_240-WT_240"]
# c_list = ["KO_15-KO_0", "KO_60-KO_15", "KO_120-KO_60", "KO_240-KO_120"]
c_list = ["WT_15-WT_0", "WT_60-WT_15", "WT_120-WT_60", "WT_240-WT_120"]
# c_string = "KO_0-WT_0"

#TRIB2 is interesting
gene = 'CISH'
if not gene:
    gene = np.random.randint(0, len(dea.data))
else:
    gene = dea.data.index.values.tolist().index(gene)
# tsplot(dea.data.iloc[gene])
idx = pd.IndexSlice
# plt.hist(dea.data.loc[:, idx['KO', :, 15, 'A']].values, bins=30)
# dea.standardize()
# tsplot(dea.data.iloc[gene])


# plt.hist(dea.data.loc[:, idx['KO', :, 15, 'A']].values, bins=30)
# plt.show()
# wt = np.mean(dea.data.loc['TUBB2B', idx['WT', :, 15, :]])
# ko = np.mean(dea.data.loc['TUBB2B', idx['KO', :, 15, :]])
# print(wt, ko, ko-wt)

# dea.suggest_contrasts()

# print(dea.expected_contrasts['KO-WT'])

dea.fit(dea.expected_contrasts['WT_ts'])
# print(dea.get_results().head())
dea.decide = dea.decide_tests(dea.fit, p_value=0.01)
# print(dea.decide.iloc[gene])
# print(dea.decide[dea.decide.any(axis=1)!=0])


# dea.fit_contrasts(dea.expected_contrasts['WT_ts'])
# print(dea.get_results().head())
# dea.decide = dea.decide_tests(dea.de_fit, p_value=0.01)
# print(dea.decide.iloc[gene])
# plt.show()
# sys.exit()

dea.cluster_trajectories()
# tsplot(dea.data.loc['ZZZ3'])
tests = dea.decide_tests(dea.fit, p_value=0.001)
# decide = dea.get_results(p_value=0.001, lfc=4)
# decide = decide[(np.sum(np.abs(tests), axis=1)!=5) & (np.sum(np.abs(tests), axis=1)!=0)]
# plt.plot(dea.times, decide.T.iloc[:len(dea.times)])
# plt.show()
# sys.exit()


dict_genes = dea.get_results(use_fstat=False)

dc = dcluster.DiscretizedClusterer()
dc.times = [0, 15, 60, 120, 240]
sets = ['WT', 'KO']
colors = ['m', 'c']
alphas = [1.0, 0.5]
paths = ['all', 'all']
fig, ax = plt.subplots(figsize=(10, 7.5))
diff = np.cumsum(tests, axis=1)
# print(diff.idxmin())
# sys.exit()
diff.insert(0, '0', 0)
diff = diff[diff.any(axis=1)!=0]
print(diff)
dc.plot_flows(ax, ['diff'], colors, alphas, paths, min_sw=0.01, max_sw=1, uniform=False, path_df=diff)
plt.show()
sys.exit()
# print(dict_genes.head())
# sys.exit()
# for g in dict_genes.index.values[:10]:
#     data = dea.data.loc[g]
#     tsplot(data)
# sys.exit()
# list_genes = dea.get_results(coef=1)
#
# dea.fit_contrasts(c_dict)
# dict_genes = dea.get_results()
# print(grepl(list_genes.index, 'WNT'))
# print(grepl(dict_genes.index, 'WNT'))
# wnts = grepl(raw_data.index.values, 'WNT')
# print(wnts)
# sys.exit()
# print(dict_genes.loc['ANGPT4'])
# for g in set(dict_genes).difference(list_genes.index.values):
#     print(g)
# sys.exit()

#
comparison = c_string
volcano_plot(dea.results, fc=1, x_colname=comparison, top_n=10, top_by=[comparison, '-log10p'], show_labels=True)
sys.exit()

# data = dea.data.iloc[random.randint(0, len(dea.data))]
# data = dea.data.loc['SPRY4']
for wnt in wnts:
    data = dea.data.loc[wnt]
# data = dea.data.loc['OTTMUSG00000000720']
    tsplot(data)

