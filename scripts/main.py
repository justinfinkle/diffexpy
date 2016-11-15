import sys, warnings
import pandas as pd
import numpy as np
from collections import Counter
from pydiffexp import DEAnalysis, volcano_plot, tsplot, filter_value
import pydiffexp.utils.multiindex_helpers as mi
import discretized_clustering as dcluster
import matplotlib.pyplot as plt

pd.set_option('display.width', 1000)

test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/data/raw_data/all_data_formatted.csv"
raw_data = pd.read_csv(test_path, index_col=0)
hierarchy = ['condition', 'well', 'time', 'replicate']

raw_data[raw_data <= 0] = 1
dea = DEAnalysis(raw_data, index_names=hierarchy, reference_labels=['condition', 'time'])

# Find differential expression at each time point
dea.fit(dea.expected_contrasts['KO-WT'])
idx = pd.IndexSlice
diffexp = dea.decide_tests(dea.de_fit)
diffexp = mi.make_hierarchical(diffexp, ['num_c', 'num_t', 'denom_c', 'denom_t', 'contrast'], split_str='-|_', keep_original=True)
dea.fit(dea.expected_contrasts['WT_ts'])
wt = mi.make_hierarchical(dea.decide_tests(dea.de_fit), ['num_c', 'num_t', 'denom_c', 'denom_t', 'contrast'], split_str='-|_', keep_original=True)
dea.fit(dea.expected_contrasts['KO_ts'])
ko = mi.make_hierarchical(dea.decide_tests(dea.de_fit), ['num_c', 'num_t', 'denom_c', 'denom_t', 'contrast'], split_str='-|_', keep_original=True)
all_tests = pd.DataFrame(pd.concat((diffexp, wt, ko), axis=1, keys=['diff', 'wt_ts', 'ko_ts']))
print(all_tests)
sys.exit()
c = np.array([diffexp.index.values, [tuple(row) for row in diffexp.values], [tuple(row) for row in wt.values],
              [tuple(row) for row in ko.values]]).T
clusters = pd.DataFrame(c, columns=['gene', 'diff', 'wt', 'ko'])
clusters.set_index(['diff', 'wt', 'ko'], inplace=True)
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