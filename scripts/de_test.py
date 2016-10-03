import sys
import pandas as pd
import de_analysis as dea
from de_analysis import DEAnalysis

# Variables
test_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/data/raw_data/all_data_formatted.csv"
raw_data = pd.read_csv(test_path).set_index('TargetID')

# Make the data into a hierarchical dataframe to initialize the differential expression object
hierarchy = ['condition', 'well', 'time', 'replicate']
h_df = dea.make_hierarchical(raw_data, index_names=hierarchy, axis=1)
de_object = DEAnalysis(h_df)


c = {'Diff0': "(KO_15-KO_0)-(WT_15-WT_0)", 'Diff15': "(KO_60-KO_15)-(WT_60-WT_15)",
     'Diff60': "(KO_120-KO_60)-(WT_120-WT_60)", 'Diff120': "(KO_240-KO_120)-(WT_240-WT_120)"}
c2 = ["KO_15-KO_0", "KO_60-KO_15", "KO_120-KO_60", "KO_240-KO_120"]
c3 = "KO_0-WT_0"

results = de_object.fit(c, p_value=0.05)
col_order = ['Diff0', 'Diff15', 'Diff60', 'Diff120'] + results.columns.tolist()[4:]
results = results[col_order]
print(results)
