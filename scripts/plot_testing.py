import sys
from pydiffexp.utils.io import read_dea_pickle
from pydiffexp import DiffExpPlot

# Load the DEAnalysis object with fits and data
dea = read_dea_pickle("./sprouty_pickle.pkl")



# Initialize a plotting object
dep = DiffExpPlot(dea)
x = dea.results['KO-WT'].top_table(coef=1)

dep.volcano_plot(x, top_n=5, show_labels=True)