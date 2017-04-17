import pandas as pd
from pydiffexp import DEAnalysis, DEPlot
import matplotlib.pyplot as plt
import sys

# Load the data
test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/sprouty/data/raw_data/GSE63497_Oncogene_Formatted.tsv"
raw_data = pd.read_csv(test_path, sep='\t', index_col=0)
hierarchy = ['condition', 'replicate']

# The example data has been background corrected, so set everything below 0 to a trivial positive value of 1
raw_data[raw_data <= 0] = 1
# Make the Differential Expression Analysis Object
# The reference labels specify how samples will be organized into unique values
dea = DEAnalysis(raw_data, index_names=hierarchy, reference_labels=['condition'], time=None, voom=True)




# Data can be standarized if desired
# norm_data = dea.standardize()

# Fit the contrasts and save the object
dea.fit_contrasts()
dep = DEPlot(dea)

# Volcano Plot
x = dea.results[0].top_table()
print(x.head())
sys.exit()
dep.volcano_plot(x, top_n=5, show_labels=True)
plt.show()
# dea.to_pickle("./sprouty_pickle.pkl")