import sys, warnings
import pandas as pd
import numpy as np
from pydiffexp import DEAnalysis, volcano_plot, tsplot, filter_value
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



