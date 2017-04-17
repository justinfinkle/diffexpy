from functools import singledispatch
import numpy as np
import pandas as pd
import sys

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

# Overloaded function to handle list of strings as well
@shannon_entropy.register(pd.DataFrame)
def _(x, axis=0, base=None):
    # Count each occurences of each unique value
    class_probability = x.apply(pd.Series.value_counts, axis=axis).fillna(0)/x.shape[axis]
    if base is None:
        base = class_probability.shape[axis]
    if class_probability.shape[axis] > 1:
        # Values with no count are set to 1 to avoid runtimerror warnings
        # Masking the array didn't seem to work.
        class_probability[class_probability == 0] = 1
        class_entropy = np.log(class_probability)/np.log(base)
        entropy = -np.sum(class_entropy * class_probability, axis=axis)
    else:
        entropy = pd.Series(np.zeros(class_probability.shape[1-axis]))

    if axis == 0:
        entropy.index = x.columns
    else:
        entropy.index = x.index
    entropy.name = 'entropy'

    return entropy


if __name__ == '__main__':
    a = np.random.randint(-1, 2, size=30)
    b = pd.DataFrame(np.random.randint(-1, 2, size=(5, 30)))
    # b = pd.DataFrame(np.zeros((5, 30)))
    print(shannon_entropy(b))
