import pandas as pd
import networkx as nx
import numpy as np


def tsv_to_dg(path):
    """
    Read a GNW gold standard tsv and return a dataframe and a DiGraph
    :param path:
    :return:
    """
    df = pd.read_csv(path, sep='\t', header=None, names=['Source', 'Target', 'Sign'])
    dg = nx.from_pandas_dataframe(df, source='Source', target='Target', create_using=nx.DiGraph())
    return df, dg


def degree_info(dg) -> pd.DataFrame:
    """
    Calculate degree information about a network
    """
    out_deg = pd.DataFrame.from_dict(dg.out_degree(), orient='index')
    in_deg = pd.DataFrame.from_dict(dg.in_degree(), orient='index')
    deg = pd.DataFrame.from_dict(dg.degree(), orient='index')
    info = pd.concat([out_deg, in_deg, deg], axis=1)                    # type: pd.DataFrame
    info.columns = ['out', 'in', 'total']
    info['(out-in)/total'] = (info.out-info['in'])/info.total

    return info


def to_gephi(df: pd.DataFrame, out_path, positive_token="+"):
    """
    Save an edge list in a gephi compatible format
    :param df:
    :param out_path:
    :param positive_token:
    :return:
    """
    tmp = df.copy()                         # type: pd.DataFrame
    # Set positive edges
    tmp[tmp.Sign == positive_token] = 1

    # Set negative edges
    tmp[tmp.Sign != positive_token] = -1
    tmp.to_csv(out_path, index=False)
    return


def make_perturbations(target, node_list, reps=3):
    """
    Create perturbations for a target node
    """
    node_idx = node_list.index(target)
    positive = np.zeros((reps, len(node_list)))
    positive[:, node_idx] = 1
    negative = positive * -1
    perturbs = np.vstack((positive, negative))
    return perturbs