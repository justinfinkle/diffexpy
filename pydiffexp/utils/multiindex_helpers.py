import warnings
import pandas as pd
from pydiffexp.utils.utils import str_convert


def is_multiindex(df):
    """
    Function to determine if a dataframe is multiindex
    :param df: dataframe
    :return: tuple
    """
    mi = [False, False]
    mi[0] = True if isinstance(df.index, pd.MultiIndex) else False
    mi[1] = True if isinstance(df.columns, pd.MultiIndex) else False
    return tuple(mi)


def make_hierarchical(df, index_names=None, split_str='_') -> pd.DataFrame:
    """
    Make a regular dataframe hierarchical by adding a MultiIndex
    :param df: dataframe; the dataframe to made hierarchical
    :param index_names: list; names for each of the categories of the multiindex
    :param axis: int (0 or 1); axis along which to split the index into a multiindex. Default (0) splits along the dataframe index, while 1 splits along the dataframe columns
    :param split_str: str; the string on which to split tuples
    :return: dataframe; hierarchical dataframe with multiindex
    """

    # Split each label into hierarchy
    try:
        index = df.columns
        s_index = split_index(index, split_str)
    except ValueError:
        df = df.T
        index = df.columns
        s_index = split_index(index, split_str)
        warnings.warn('Multiindex found for rows, but not columns. Returned data frame is transposed from input')

    h_df = df.copy()
    m_index = pd.MultiIndex.from_tuples(s_index, names=index_names)
    h_df.columns = m_index

    return h_df


def split_index(index, split_str):
    """
    Split a list of strings into a list of tuples.
    :param index: list-like; List of strings to be split
    :param split_str: str; substring by which to split each string
    :return:
    """
    s_index = [tuple(map(str_convert, ind.split(split_str))) for ind in index if split_str in ind]
    if len(s_index) != len(index):
        raise ValueError('Index not split properly using supplied string')
    return s_index