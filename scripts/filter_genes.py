import sys
import pandas as pd
import numpy as np
from pydiffexp.utils.io import read_dea_pickle
from pydiffexp.utils.utils import column_unique
import matplotlib.pyplot as plt
from pydiffexp import DiffExpPlot

pd.set_option('display.width', 1000)

# Load the DEAnalysis object with fits and data
dea = read_dea_pickle("./sprouty_pickle.pkl")

# ======================================================================================================================
# ======================================================================================================================
# Genes that are initially DE
# ======================================================================================================================
# ======================================================================================================================

# Useful params
p_val = 0.05
ic = 'KO_0-WT_0'    # Initial contrast
der = dea.results['KO-WT']

# This represents traditional analysis because it only considers the first time point
de_ko = der.top_table(coef=1, p=p_val)
print("%i genes DE initially\n" % len(de_ko), de_ko.head())


# If we consider additional timepoints the results change slightly because of the global multiple hypothesis correction
de_ko_global = der.discrete.loc[:, ic]
de = de_ko_global!=0
print('\n', len(de_ko_global[de]), 'DE when considering global effects')

# Genes that start off differentially expressed but no longer are
converge = der.discrete[(de) & (der.discrete==0).any(axis=1)]
print(der.count_clusters(der.cluster_discrete(converge)).sort_index())
print(der.cluster_discrete(converge)[der.cluster_discrete(converge)['Cluster'] == '(-1, -1, 0, 0, 0)'])

dep = DiffExpPlot(dea)

x = dea.data.loc['TIAM2']
dep.tsplot(x)
plt.show()




