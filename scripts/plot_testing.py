import sys
from pydiffexp.utils.io import read_dea_pickle
from pydiffexp import DiffExpPlot

# Load the DEAnalysis object with fits and data
dea = read_dea_pickle("./sprouty_pickle.pkl")

print(dea.data)
sys.exit()

# Initialize a plotting object
dep = DiffExpPlot(dea)
