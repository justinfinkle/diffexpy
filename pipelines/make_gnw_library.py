import itertools as it
import multiprocessing as mp
import sys

import networkx as nx
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
    # Calculate edges in
    edge_in = net_structure.abs().groupby(level=1)['edge_sign'].sum()
    targets, n_in, logics = zip(*[(target, num_in, ["{}_linear".format(target), "{}_multiplicative".format(target)])
                                  if num_in>1 else (target, num_in, target)
              for target, num_in in edge_in.iteritems()])

    # Calculate combos of input logics
    combos = pd.DataFrame(list(it.product(*logics)), columns=["{}_logic".format(t) for t in targets])

    # Create meta data for the network structure
    labels = ["{}_in".format(t) for t in targets] + net_structure.index.get_level_values('edge').values.tolist()
    net_data = pd.DataFrame([list(n_in) + net_structure['edge_sign'].values.tolist()] * len(combos), columns=labels)

    # Combine information
    return pd.concat([combos, net_data], axis=1)


def check_weakly_connected(series):
    series = series[1]
    tmp_df = series.reset_index()
    tmp_df['sign'] = series.values
    delete_edges = tmp_df[tmp_df['sign'] == 0]
    dg = nx.from_pandas_dataframe(tmp_df, source='source',
                                          target='target', edge_attr=['sign'], create_using=nx.DiGraph())
    dg.remove_edges_from(delete_edges.edge.values.tolist())

    return nx.is_weakly_connected(dg)


def drop_nets(df, signs):
    return df.iloc[:, signs.shape[1]:].abs().sum() >=1


def edge_combos(possible_edges, weakly_connected=True):
    # Calculate combinations of boolean edge values
    nodes = set(possible_edges.source)
    edge_values = pd.DataFrame(np.array(list(it.product([0, 1, -1], repeat=len(possible_edges)))).T)
    labels = possible_edges.columns.values.tolist()

    if weakly_connected:
        # Simple filtering: number of edges must be >= n_nodes-1
        edge_values = edge_values.loc[:, edge_values.abs().sum()>=(len(nodes)-1)]

    df = pd.concat([possible_edges, edge_values], axis=1)               # type: pd.DataFrame

    if weakly_connected:
        # Find which node is a target and source in each network
        node_as_target = df.groupby('target').apply(drop_nets, signs=possible_edges)
        node_as_source = df.groupby('source').apply(drop_nets, signs=possible_edges)

        df.set_index(labels, inplace=True)

        # Remove networks for which each node is not either a target or source (i.e. it is orphaned)
        df = df.loc[:, (node_as_target | node_as_source).sum() == len(nodes)]

        # Remove any other networks that might have missed filtering
        pool = mp.Pool()
        d = pool.map(check_weakly_connected, list(df.iteritems()))
        pool.close()
        pool.join()
        df = df.loc[:, d]
        # Reset the columns and index
        df.columns = range(0, df.shape[1])
        df.reset_index(inplace=True)

    # Melt the dataframe to a workable shape
    melted = pd.melt(df.reset_index(), id_vars=labels,
                     value_vars=list(range(0, df.shape[1]-len(labels))), var_name='net_id', value_name='edge_sign')

    melted.set_index(labels, inplace=True)
    # Group by network
    network_stats = melted.groupby(['net_id'])

    library_info = network_stats.apply(logic_combos) # type: pd.DataFrame
    library_info.index.set_names('combo', level=1, inplace=True)
    return library_info.reset_index()


if __name__ == '__main__':
    pd.set_option('display.width', 250)
    nodes = nodes = np.array(['x', 'G', 'y'])
    edge_df = make_possible_edge_list(nodes, nodes, self_edges=False)

    # Create the library of networks that needs to be created
    library = edge_combos(edge_df)

    # Simulate the networks
    for i, info in library.iterrows():
        print(i)
        print(info)
        sys.exit()
