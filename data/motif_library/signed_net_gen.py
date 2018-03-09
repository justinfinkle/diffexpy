import pandas as pd
import numpy as np
import itertools as it
import time
import networkx as nx
import functools
import operator
import networkx.algorithms.isomorphism as iso


def count_combos(dg):
    combos = {0: 1, 1: 1, 2: 2}
    n_in = [combos[len(dg.in_edges(n))] for n in dg.nodes()]

    return functools.reduce(operator.mul, n_in, 1)


def make_possible_edge_list(parents, children, self_edges=True):
    """
    Create a list of all the possible edges between parents and children

    :param parents: array
        labels for parents
    :param children: array
        labels for children
    :param self_edges:
    :return: array, length = parents * children
        array of parent, child combinations for all possible edges
    """
    parent_index = range(len(parents))
    child_index = range(len(children))

    a, b = np.meshgrid(parent_index, child_index)
    parent_list = list(parents[a.flatten()])
    child_list = list(children[b.flatten()])
    possible_edge_list = None
    if self_edges:
        possible_edge_list = list(zip(parent_list, child_list))

    elif not self_edges:
        possible_edge_list = [x for x in zip(parent_list, child_list) if x[0] != x[1]]

    return possible_edge_list

# Define nodes
nodes = np.array(['x', 'G', 'y'])

# Get possible edges
edges = make_possible_edge_list(nodes, nodes, self_edges=False)
df = pd.DataFrame(edges, columns=['source', 'target'])
df['edge'] = edges

# Add/remove -1 to make signed/unsigned
edge_values = list(it.product([0, 1], repeat=len(edges)))

# Edge matching function
em = iso.numerical_edge_match('sign', 0)

# Initialize unique graph dictionary for storage and time variables for tracking
unique_graphs = {n_edges: [] for n_edges in range(len(nodes) - 1, len(edges) + 2)}
start = time.time()
int_time = start

combos = []

# Generate networks
for ii, ev in enumerate(edge_values):
    if ii % 1000 == 0:
        print(ii/len(edge_values)*100, '% done |',
              'since start:', time.time()-start,
              '| since last print:', time.time()-int_time)
        int_time = time.time()

    tmp_df = df.copy()
    tmp_df['sign'] = ev

    # Drop edges that don't exsits and recolor based on sign
    tmp_df = tmp_df[tmp_df.sign != 0]
    tmp_df['color'] = 'green'
    tmp_df.loc[tmp_df.sign == -1, 'color'] = 'red'

    # Make networkx directed graph
    current_dg = nx.from_pandas_dataframe(tmp_df, source='source',
                                          target='target', edge_attr=['sign', 'color'], create_using=nx.DiGraph())
    # Add any left out nodes
    current_dg.add_nodes_from(nodes)

    n_dg_edges = len(current_dg.edges())

    # Confirm is weakly connected
    if not nx.is_weakly_connected(current_dg):
        continue

    # Because the node name matters, don't need to check for isomorphisms
    unique_graphs[n_dg_edges].append(current_dg)
    combos.append(count_combos(current_dg))

print('# Graphs: ', len(combos))
print('# Graphs with Complex Interactions:', sum(combos))
pd.to_pickle(unique_graphs, './unique_wc_3node_unsigned_noself_nets.pkl')
