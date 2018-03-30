import itertools as it

import numpy as np
import pandas as pd


def make_possible_edge_list(parents, children, self_edges=True):
    """
    Create a list of all the possible edges between parents and children

    :param parents: array
        labels for parents
    :param children: array
        labels for children
    :param self_edges:
    :return: df, length = parents * children
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

    df = pd.DataFrame(possible_edge_list, columns=['source', 'target'])
    df['edge'] = possible_edge_list

    return df


def logic_combos(net_structure):
    print(net_structure)
    # print(net_structure[net_structure.edge_sign != 0])
    edge_in = net_structure.abs().groupby(level=1)['edge_sign'].sum()
    targets, n_in, logics = zip(*[(target, num_in, ["{}_linear".format(target), "{}_multiplicative".format(target)]) if num_in>1 else (target, num_in, target)
              for target, num_in in edge_in.iteritems()])
    print(list(targets))
    print(list(it.product(*logics)))
    print(n_in)
    print(pd.DataFrame(list(it.product(*logics)), columns=list(targets)))


def edge_combos(possible_edges, weakly_connected=True):
    # Calculate combinations of boolean edge values
    nodes = set(possible_edges.source)
    edge_values = pd.DataFrame(np.array(list(it.product([0, 1, -1], repeat=len(possible_edges)))).T)
    if weakly_connected:
        # Simple filtering: number of edges must be >= n_nodes-1
        edge_values = edge_values.loc[:, edge_values.abs().sum()>=(len(nodes)-1)]

    df = pd.concat([possible_edges, edge_values], axis=1)               # type: pd.DataFrame
    # Melt the dataframe to a workable shape
    melted = pd.melt(df.reset_index(), id_vars=['source', 'target', 'edge'],
                     value_vars=list(range(0, edge_values.shape[1])), var_name='net_id', value_name='edge_sign')

    melted.set_index(['source', 'target', 'edge'], inplace=True)
    print(len(set(melted['net_id'])))
    # Group by network
    network_stats = melted.groupby(['net_id'])
    logic_combos(network_stats.get_group(0))
    return melted


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    nodes = nodes = np.array(['x', 'G', 'y'])
    edge_df = make_possible_edge_list(nodes, nodes, self_edges=False)
    edge_combos(edge_df)
