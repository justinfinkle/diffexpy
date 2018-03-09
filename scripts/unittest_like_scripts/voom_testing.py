import sys

import pandas as pd
from pydiffexp import DEAnalysis, DEPlot

pd.set_option('display.width', 1000)

# Load the data
test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/sprouty/data/raw_data/GSE63497_Oncogene_Formatted.tsv"
# test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/sprouty/data/raw_data/GSE63497_VEC_CRE_Formatted.tsv"
raw_data = pd.read_csv(test_path, sep='\t', index_col=0)
hierarchy = ['condition', 'replicate']

# The example data has been background corrected, so set everything below 0 to a trivial positive value of 1
raw_data[raw_data <= 0] = 1

# Remove all genes with low counts so voom isn't confused
raw_data = raw_data[~(raw_data < 5).all(axis=1)]
# Make the Differential Expression Analysis Object
# The reference labels specify how samples will be organized into unique values
dea = DEAnalysis(raw_data, index_names=hierarchy, reference_labels=['condition'], time=None, voom=True)
# Data can be standarized if desired
# norm_data = dea.standardize()

# Fit the contrasts and save the object
# cont = dea.possible_contrasts()
# cont[0] = 'CRE-BRaf'
dea.fit_contrasts()
dep = DEPlot(dea)
sys.exit()
# Volcano Plot
x = dea.results[0].top_table(p=0.05)

# sns.clustermap(x.iloc[:, :10])
genes = utils.grepl('SOX', x.index)
g = sns.clustermap(x.loc[genes].iloc[:, :10])
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=30)
plt.show()
sys.exit()
gene = 'SPRY4'
print(rh.rvect_to_py(dea.data_matrix).loc[gene].reset_index())
print(dea.data.loc[gene])
# ax = sns.boxplot(data=rh.rvect_to_py(dea.data_matrix).loc[gene].reset_index(), x='index', y=gene)
ax = sns.swarmplot(data=rh.rvect_to_py(dea.data_matrix).loc[gene].reset_index(), x='index', y=gene, size=10)
plt.xlabel('Condition', fontsize=20, fontweight='bold')
plt.ylabel(('%s Estimated log2 CPM' % gene), fontsize=20, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.show()
sys.exit()
dep.volcano_plot(x, top_n=5, show_labels=True, top_by=['-log10p', 'logFC'])
plt.tight_layout()
plt.show()
# dea.to_pickle("./sprouty_pickle.pkl")