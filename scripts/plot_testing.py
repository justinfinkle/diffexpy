import sys
from pydiffexp.utils.io import read_dea_pickle
from pydiffexp import DiffExpPlot

# Load the DEAnalysis object with fits and data
dea = read_dea_pickle("./sprouty_pickle.pkl")



# Initialize a plotting object
dep = DiffExpPlot(dea)
x = dea.results['KO-WT'].top_table(coef=1, use_fstat=False)

dep.volcano_plot(x)