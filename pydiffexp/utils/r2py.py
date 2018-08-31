import numpy as np
import pandas as pd
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


def pydf_to_rmat(x) -> robj.vectors.Matrix:
    """
    Convert a pandas dataframe to an R matrix
    :param x:
    :return:
    """
    r_matrix = robj.r.matrix(x.values, nrow=x.shape[0], ncol=x.shape[1])
    r_matrix.colnames = robj.StrVector(x.columns.values)
    r_matrix.rownames = robj.StrVector(x.index.values)

    return r_matrix


def rvect_to_py(vector, force_list=False):
    """
    Convert an R vector to its appropriate python equivalent
    :param vector:
    :param force_list: bool; force the output to be a list. Default (False) returns lists only if len > 1
    :return:
    """
    x = None

    # DataFrame
    if isinstance(vector, robj.vectors.DataFrame):
        x = rdf_to_pydf(vector)

    # Matrix
    elif isinstance(vector, robj.vectors.Matrix):
        x = pd.DataFrame(np.array(vector))

        # Correct if row and column names do not exist
        if vector.rownames != robj.NULL:
            x.index = rvect_to_py(vector.rownames, force_list=True)
        if vector.colnames != robj.NULL:
            x.columns = rvect_to_py(vector.colnames, force_list=True)


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

    # Bools
    elif isinstance(vector, robj.vectors.BoolVector):
        x = np.array(vector).astype(bool)

    else:
        raise TypeError('R data type not recognized')

    # If it is an array with just one value, unpack that (e.g. Str, Int, and Float)
    if x is not None:
        if isinstance(x, np.ndarray) & (len(x) == 1) & ~force_list:
            x = x[0]

    return x
