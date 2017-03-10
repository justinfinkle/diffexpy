"""
This is a basic example that demonstrates how to load data, do a fit
"""

import pandas as pd
from pydiffexp import DEAnalysis

# Load the data
test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/sprouty/data/raw_data/all_data_formatted.csv"
raw_data = pd.read_csv(test_path, index_col=0)
hierarchy = ['condition', 'well', 'time', 'replicate']

# The example data has been background corrected, so set everything below 0 to a trivial positive value of 1
raw_data[raw_data <= 0] = 1

# Make the Differential Expression Analysis Object
# The reference labels specify how samples will be organized into unique values
dea = DEAnalysis(raw_data, index_names=hierarchy, reference_labels=['condition', 'time'])

# Data can be standarized if desired
# norm_data = dea.standardize()

# Fit the contrasts and save the object
dea.fit_contrasts()
print(dea.results['KO_ar'].continuous.loc['FAM110C'])
# dea.to_pickle("./sprouty_pickle.pkl")

