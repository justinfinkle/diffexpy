import ast

import pandas as pd

dea = pd.read_pickle('../intermediate_data/strongly_connected_dea.pkl')
der = dea.results['ko-wt']
scores = der.score_clustering()

# Remove clusters that have no dynamic DE (i.e. all 1, -1, 0)
interesting = scores.loc[scores.Cluster.apply(ast.literal_eval).apply(set).apply(len) > 1]
print(interesting)
