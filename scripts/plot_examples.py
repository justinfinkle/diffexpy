import sys, warnings
import pandas as pd
import random
from pydiffexp import DEAnalysis, volcano_plot, tsplot


def grepl(search_list, substr):
    grep_list = list(filter(lambda x: substr in x, search_list))
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
c_list = ["KO_15-KO_0", "KO_60-KO_15", "KO_120-KO_60", "KO_240-KO_120"]
c_string = "KO_0-WT_0"

dea.fit(c_list)

dict_genes = dea.get_results(use_fstat=False)
print(sum(dea.decide.all(axis=1) != 0))
sys.exit()
# print(dict_genes.head())
# sys.exit()
# for g in dict_genes.index.values[:10]:
#     data = dea.data.loc[g]
#     tsplot(data)
# sys.exit()
# list_genes = dea.get_results(coef=1)
#
# dea.fit(c_dict)
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

