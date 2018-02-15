import sys
import pandas as pd
from pydiffexp import DEAnalysis, DEPlot
import matplotlib.pyplot as plt

# Load the data
data_path = "/Volumes/Hephaestus/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/GSE13100/" \
            "log2_bgcorrected_GSE13100_RA_data.pkl"
raw_data = pd.read_pickle(data_path)

# The example data has been background corrected, so set everything below 0 to a trivial positive value of 1
raw_data[raw_data <= 0] = 1

# Make the Differential Expression Analysis Object
# The reference labels specify how samples will be organized into unique values
dea = DEAnalysis(raw_data, replicate='rep', reference_labels=['condition', 'time'])

# Data can be standarized if desired
# norm_data = dea.standardize()

# Fit the contrasts and save the object
dea.fit_contrasts()
# print(dea.results['MUT-WT'].top_table(p=0.05))

dep = DEPlot(dea)
x = dea.results['MUT-WT'].top_table(coef=7, use_fstat=False)
dep.volcano_plot(x, top_n=5, show_labels=True)
plt.show()