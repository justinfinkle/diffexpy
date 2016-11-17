import sys
import pandas as pd
import numpy as np
import rpy2.robjects as robj
from rpy2.robjects.methods import RS4


class MArrayLM(RS4):
    """
    Class to wrap MArrayLM from R. Makes data more easily accessible
    """
    def __init__(self, obj):
        """

        :param obj:
        """
        # Store the original object
        self.robj = obj                 # type: robj.vectors.ListVector

        # Initialize expected attributes. See R documentation on MArrayLM for more details on attributes
        self.Amean = None               # type: np.ndarray
        self.F = None                   # type: np.ndarray
        self.F_p_value = None           # type: np.ndarray
        self.assign = None              # type: np.ndarray
        self.coefficients = None        # type: pd.DataFrame
        self.contrasts = None           # type: pd.DataFrame
        self.cov_coefficients = None    # type: pd.DataFrame
        self.design = None              # type: pd.DataFrame
        self.df_prior = None            # type: float
        self.df_residual = None         # type: np.ndarray
        self.df_total = None            # type: np.ndarray
        self.lods = None                # type: pd.DataFrame
        self.method = None              # type: str
        self.p_value = None             # type: pd.DataFrame
        self.proportion = None          # type: float
        self.qr = None                  # type: dict
        self.rank = None                # type: int
        self.s2_post = None             # type: np.ndarray
        self.s2_prior = None            # type: float
        self.sigma = None               # type: np.ndarray
        self.stdev_unscaled = None      # type: pd.DataFrame
        self.t = None                   # type: pd.DataFrame
        self.var_prior = None           # type: float

        # Unpact the values
        self.unpack()

    def unpack(self):
        """
        Unpack the MArrayLM object (rpy2 listvector) into an object.
        :return:
        """
        # Unpack the list vector object
        data = unpack_r_listvector(self.robj)

        # Store the values into attributes
        for k, v in data.items():
            setattr(self, k, v)


def unpack_r_listvector(l_vector: robj.vectors.ListVector) -> dict:
    """
    Unpack a list vector. Can be used recursively
    :param l_vector:
    :return:
    """
    d = {name.replace('.', '_'): rvect_to_py(value) for name, value in zip(l_vector.names, l_vector)}
    return d


def rvect_to_py(vector):
    """
    Convert an R vector to its appropriate python equivalent
    :param vector:
    :return:
    """
    x = None

    # Matrix
    if isinstance(vector, robj.vectors.Matrix):
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
