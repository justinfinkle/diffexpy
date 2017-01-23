import sys
from pydiffexp.utils.io import read_dea_pickle
from pydiffexp import DiffExpPlot
import matplotlib.pyplot as plt

# Load the DEAnalysis object with fits and data
dea = read_dea_pickle("./sprouty_pickle.pkl")

# Initialize a plotting object
dep = DiffExpPlot(dea)

# Volcano Plot
x = dea.results['KO-WT'].top_table(coef=1, use_fstat=False)
# dep.volcano_plot(x, top_n=5, show_labels=True)

# Time Series Plot
x = dea.data.loc['CISH']

dep.tsplot(x)
plt.show()

#
