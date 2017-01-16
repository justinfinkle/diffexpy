from pydiffexp import DEAnalysis, DEResults
import pandas as pd


def read_dea_pickle(path) -> DEAnalysis:
    """
    Load a pickle as a DEAanlysis object
    :param path: pickle path
    :return:
    """

    return pd.read_pickle(path)


def read_der_pickle(path) -> DEResults:
    """
    Load a pickle as a DEAanlysis object
    :param path: pickle path
    :return:
    """

    return pd.read_pickle(path)
