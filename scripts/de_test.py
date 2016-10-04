import sys
import pandas as pd
from pydiffexp import DEAnalysis, make_hierarchical, is_multiindex

# Variables
test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/data/raw_data/all_data_formatted.csv"
raw_data = pd.read_csv(test_path, index_col=0)

dea = DEAnalysis()
# Make the data into a hierarchical dataframe to initialize the differential expression object
hierarchy = ['condition', 'well', 'time', 'replicate']
dea.set_data(raw_data.T, index_names=hierarchy)
print(dea.data.head())
sys.exit()

# Types of contrasts
c_dict = {'Diff0': "(KO_15-KO_0)-(WT_15-WT_0)", 'Diff15': "(KO_60-KO_15)-(WT_60-WT_15)",
          'Diff60': "(KO_120-KO_60)-(WT_120-WT_60)", 'Diff120': "(KO_240-KO_120)-(WT_240-WT_120)"}
c_list = ["KO_15-KO_0", "KO_60-KO_15", "KO_120-KO_60", "KO_240-KO_120"]
c_string = "KO_0-WT_0"

results = dea.fit(c_string, p_value=0.05)
# col_order = ['Diff0', 'Diff15', 'Diff60', 'Diff120'] + results.columns.tolist()[4:]
print(results)

