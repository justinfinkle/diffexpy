import sys
from pydiffexp.utils.io import read_dea_pickle


dea = read_dea_pickle("./sprouty_pickle.pkl")
print(dea.results)
sys.exit()
