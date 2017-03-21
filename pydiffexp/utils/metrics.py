from functools import singledispatch
import numpy as np
import math
from sklearn.metrics import mutual_info_score

@singledispatch
def shannon_entropy(x, base=None):
    """
    Calculate shannon entropy of a list like with discrete classes
    :param x:
    :param base: int; which base should be used for calculating the units
    :return:
    """

    unique = set(x)
    if base is None:
        base = len(unique)

    if len(unique) > 1:
        class_probability = np.array([len(np.where(x==label)[0]) for label in unique])/len(x)

        # Use the log base change rule to make numpy work
        class_entropy = np.log(class_probability)/np.log(base)
        entropy = -np.sum(class_entropy * class_probability)
    else:
        entropy = 0

    return entropy


if __name__ == '__main__':
    a = np.random.randint(-1, 2, size=30)
    print(shannon_entropy(a))
