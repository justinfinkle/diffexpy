import sys, warnings
import pandas as pd
import numpy as np
from collections import Counter
from pydiffexp import DEAnalysis, volcano_plot, tsplot, filter_value
import pydiffexp.utils.multiindex_helpers as mi
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
dea.fit(dea.expected_contrasts['KO-WT'])
res = dea.get_results(coef=1, p_value=1)
res = res[res['pval'] <= 0.05].loc['GBF1']
gene = 'GBF1'
idx = pd.IndexSlice
ps = []
gs = []
for ci in np.linspace(0,1):
    cis = []
    gs = []
    ps = []
    for i in [0, 15, 60, 120, 240]:
        wt = dea.data.loc[gene, idx['WT', :, i, :]]
        ko = dea.data.loc[gene, idx['KO', :, i, :]]
        x_wt = np.mean(wt)
        x_ko = np.mean(ko)
        desired_p = 0.031826709456797622
        pooled_df = len(wt)+len(ko) - 2
        t_sig = stats.t.ppf(1-desired_p/2, df=pooled_df)
        percent_ci = ci
        t_ci_wt = stats.t.ppf(1-(1-percent_ci)/2, df=len(wt)-1)
        t_ci_ko = stats.t.ppf(1-(1-percent_ci)/2, df=len(ko)-1)
        wt_err = t_ci_wt*stats.sem(wt)
        ko_err = t_ci_ko*stats.sem(ko)
        pval = stats.ttest_ind(wt, ko, equal_var=True)[1]
        # print(stats.sem(wt), np.std(wt, ddof=1), np.sqrt(len(wt)), np.std(wt, ddof=1)/np.sqrt(len(wt)))
        # print(pooled_df, t_sig, t_ci_wt, t_ci_ko)
        # plt.errorbar([1,1], [x_wt, x_ko], fmt='o', yerr=[wt_err, ko_err])
        gap = np.abs(x_wt-x_ko)-(wt_err+ko_err)
        ps.append(pval)
        gs.append(gap)
        cis.append(ci)
    plt.plot(ps, gs, 'o')
plt.show()
sys.exit()
print(stats.t.cdf(np.abs(x_wt-x_ko)/(stats.sem(wt)+stats.sem(ko)), df=len(wt)-1))
f = ((stats.sem(wt)+stats.sem(ko))/np.abs(x_wt-x_ko))
dp = (1-stats.t.cdf(1/f, df=len(wt)-1))/2
t_sig = stats.t.ppf(1-dp/2, df=pooled_df)
sys.exit()
# print(dea.get_results(coef=1, n=5))
# print(dea.de_fit[17])
# sys.exit()
idx = pd.IndexSlice
diffexp = dea.decide_tests(dea.de_fit)
diffexp = mi.make_hierarchical(diffexp, ['num_c', 'num_t', 'denom_c', 'denom_t', 'original'], split_str='-|_', keep_original=True)
dea.fit(dea.expected_contrasts['WT_ts'])
wt = mi.make_hierarchical(dea.decide_tests(dea.de_fit), ['num_c', 'num_t', 'denom_c', 'denom_t', 'original'], split_str='-|_', keep_original=True)
dea.fit(dea.expected_contrasts['KO_ts'])
ko = mi.make_hierarchical(dea.decide_tests(dea.de_fit), ['num_c', 'num_t', 'denom_c', 'denom_t', 'original'], split_str='-|_', keep_original=True)
all_tests = pd.DataFrame(pd.concat((diffexp, wt, ko), axis=1, keys=['diff', 'wt_ts', 'ko_ts'], names=['contrasts']+ko.columns.names))
c = np.array([diffexp.index.values, [tuple(row) for row in diffexp.values], [tuple(row) for row in wt.values],
              [tuple(row) for row in ko.values]]).T
clusters = pd.DataFrame(c, columns=['gene', 'diff', 'wt', 'ko'])
gene = 'BTD'

clusters.set_index(['diff', 'wt', 'ko'], inplace=True)
clusters.sort_index(inplace=True)
print(clusters.loc[idx[(0,0,1,1,1)]])
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