__author__ = 'Justin Finkle'
__email__ = 'jfinkle@u.northwestern.edu'

import sys
import network_analysis as na
import pandas as pd
import numpy as np
import networkx as nx

# Load link list
ll = pd.DataFrame(pd.read_csv('../network_inference/20150908_dionesus_log_fc_intersection_linked_list.csv'))
ll.drop(ll.columns[0], axis=1, inplace=True)

edge_list = zip(ll.Parent, ll.Child)
dg = nx.DiGraph()
dg.add_edges_from(edge_list)
cutoff=2
paths = nx.all_simple_paths(dg, 'EGR1', 'SPRY2', cutoff)
n_paths = 0
for path in paths:
    if n_paths<100:
        print path
    n_paths+=1

print n_paths