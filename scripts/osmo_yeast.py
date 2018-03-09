import pandas as pd
from pydiffexp import DEAnalysis

# Load the data
data_path = "/Volumes/Hephaestus/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/GSE13100/" \
            "bgcorrected_GSE13100_TR_data.pkl"
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
dea.to_pickle('intermediate_data/yeast_osmoTR_dea.pkl')