import sys, warnings
import pandas as pd
import numpy as np
from collections import Counter
from pydiffexp import DEAnalysis, volcano_plot, tsplot
import pydiffexp.utils.multiindex_helpers as mi
import pydiffexp.utils.rpy2_helpers as rh
import discretized_clustering as dcluster
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('display.width', 1000)

test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/data/raw_data/all_data_formatted.csv"
raw_data = pd.read_csv(test_path, index_col=0)
hierarchy = ['condition', 'well', 'time', 'replicate']

raw_data[raw_data <= 0] = 1
dea = DEAnalysis(raw_data, index_names=hierarchy, reference_labels=['condition', 'time'])


# Find differential expression at each time point
gene = 'TUBB2B'

# dea.fit_contrasts(dea.expected_contrasts['KO-WT'])
# print(dea.get_results())
# print(dea.decide_tests(dea.fit).loc[gene])

dea.fit_contrasts([dea.expected_contrasts['KO_ar-WT_ar'], dea.expected_contrasts['KO_ts']], names=['AR', 'KO_ts'])
print(dea.fit)
sys.exit()
print(dea.get_results(n=5))
print(dea.decide_tests(dea.fit).loc[gene])

dea.fit_contrasts(dea.expected_contrasts['KO_ar'])
# r = dea.get_results()
# plt.plot(dea.times[1:], r.iloc[:100, :4].T)
print(dea.decide_tests(dea.fit).loc[gene])
# plt.legend()

dea.fit_contrasts(dea.expected_contrasts['WT_ar'])
# r = dea.get_results()
# plt.plot(dea.times[1:], r.iloc[:100, :4].T)
print(dea.decide_tests(dea.fit).loc[gene])

tsplot(dea.data.loc[gene])
plt.show()

sys.exit()
# sys.exit()

idx = pd.IndexSlice
diffexp = dea.decide_tests(dea.fit)
diffexp = mi.make_hierarchical(diffexp, ['num_c', 'num_t', 'denom_c', 'denom_t', 'original'], split_str='-|_', keep_original=True)
dea.fit_contrasts(dea.expected_contrasts['WT_ts'])
wt = mi.make_hierarchical(dea.decide_tests(dea.fit), ['num_c', 'num_t', 'denom_c', 'denom_t', 'original'], split_str='-|_', keep_original=True)
dea.fit_contrasts(dea.expected_contrasts['KO_ts'])
ko = mi.make_hierarchical(dea.decide_tests(dea.fit), ['num_c', 'num_t', 'denom_c', 'denom_t', 'original'], split_str='-|_', keep_original=True)
all_tests = pd.DataFrame(pd.concat((diffexp, wt, ko), axis=1, keys=['diff', 'wt_ts', 'ko_ts'], names=['contrasts']+ko.columns.names))
c = np.array([diffexp.index.values, [tuple(row) for row in diffexp.values], [tuple(row) for row in wt.values],
              [tuple(row) for row in ko.values]]).T
clusters = pd.DataFrame(c, columns=['gene', 'diff', 'wt', 'ko'])
gene = 'AREG'

clusters.set_index(['diff', 'wt', 'ko'], inplace=True)
clusters.sort_index(inplace=True)
print(clusters.loc[idx[(0,0,1,1,1)]])
plt.figure()
tsplot(dea.data.loc[gene])
g = dea.data.loc[gene, idx['KO']]
g = g.reset_index()
g = g.groupby('time')
g_ko = np.array([[g, np.mean(data[gene])] for g, data in g]).T


g = dea.data.loc[gene, idx['WT']]
g = g.reset_index()
g = g.groupby('time')
g_wt = np.array([[g, np.mean(data[gene])] for g, data in g]).T
plt.plot(g_ko[0], g_ko[1]-g_wt[1])
plt.show()
sys.exit()
tsplot(dea.data.loc['FEZ2'])
plt.show()
sys.exit()
counts = clusters.groupby(level=[1]).count()
counts.drop((0,0,0,0), inplace=True)
plt.bar(range(len(counts)), counts.values)
plt.xticks(range(len(counts)), ["_".join([str(x) for x in i]) for i in counts.index.values], rotation='vertical')
plt.tight_layout()
plt.show()
sys.exit()
nozeros = diffexp[(np.sum(np.abs(diffexp), axis=1)) == 5]
n_changes = np.abs(np.sum(nozeros, axis=1))
same = nozeros[n_changes == 5]
print(len(nozeros), len(same))