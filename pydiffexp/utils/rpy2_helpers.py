import pandas as pd
import numpy as np
import rpy2.robjects as robj
from rpy2.robjects import pandas2ri


def unpack_r_listvector(l_vector: robj.vectors.ListVector) -> dict:
    """
    Unpack a list vector. Can be used recursively
    :param l_vector:
    :return:
    """
    d = {name.replace('.', '_'): rvect_to_py(value) for name, value in zip(l_vector.names, l_vector)}
    return d


def rdf_to_pydf(x):
    """Convert an R dataframe to a python dataframe"""
    '''
    The converter is activated and then deactivated. There have been some reports of inconsistencies if the
    converter is activated during import
    '''
    pandas2ri.activate()
    df = pandas2ri.ri2py(x)
    pandas2ri.deactivate()
    return df


def rvect_to_py(vector):
    """
    Convert an R vector to its appropriate python equivalent
    :param vector:
    :return:
    """
    x = None

    # DataFrame
    if isinstance(vector, robj.vectors.DataFrame):
        x = rdf_to_pydf(vector)

    # Matrix
    elif isinstance(vector, robj.vectors.Matrix):
        x = pd.DataFrame(np.array(vector), index=vector.rownames, columns=vector.colnames)

    # Integers
    elif isinstance(vector, robj.vectors.IntVector):
        x = np.array(vector).astype(int)

    # Floats
    elif isinstance(vector, robj.vectors.FloatVector):
        x = np.array(vector)

    # List - will be called recursively
    elif isinstance(vector, robj.vectors.ListVector):
        x = unpack_r_listvector(vector)

    # Strings
    elif isinstance(vector, robj.vectors.StrVector):
        x = np.array(vector).astype(str)

    # If it is an array with just one value, unpack that (e.g. Str, Int, and Float)
    if isinstance(x, np.ndarray) & len(x) == 1:
        x = x[0]

    return x
