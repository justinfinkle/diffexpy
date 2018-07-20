from functools import singledispatch
import pandas as pd
from itertools import chain
import string

@singledispatch
def grepl(substr, search_list):
    grep_list = list(filter(lambda x: substr in x, search_list))
    return grep_list


# Overloaded function to handle list of strings as well
@grepl.register(list)
def _(str_list, search_list, flat=True):
    grep_list = [grepl(substr, search_list) for substr in str_list]

    # Flatten the list of lists
    if flat:
        grep_list = [item for sublist in grep_list for item in sublist]

    return grep_list


def str_convert(s):
    try:
        s = int_or_float(s)
    except ValueError:
        pass
    return s


def contrast_map(x, t1, t2=None):
    # x is a tuple of conditions
    x = list(map(str, x))
    if t2 is None:
        t2 = t1
    return x[0]+"_"+str(t1)+"-"+x[1]+"_"+str(t2)


def int_or_float(s):
    # Note: this is probably not a safe method for complex inputs
    r = float(s) if int(s) < float(s) else int(s)
    return r


def filter_value(x, value, axis=1, criteria='all'):
    if criteria == 'all':
        x = x[~(x == value).all(axis=axis)]

    elif criteria == 'any':
        x = x[~(x == value).any(axis=axis)]

    return x


def column_unique(x):
    """
    Count the number of unique values in each column
    :param df:
    :return:
    """
    df = pd.DataFrame()
    for col in x:
        data = x[col]
        df[data.name] = data.value_counts()
    return df


def all_subsets(groups, labels=None):
    """
    Find all subsets of groups

    Adapted from get_labels in venn diagram package
    https://github.com/tctianchi/pyvenn/blob/master/venn.py

    :param groups: iterable of iterables
    :return:
    """
    if labels is None:
        labels = list(string.ascii_uppercase[:len(groups)])

    n = len(groups)

    # Make all groupings into sets
    sets_data = [set(g) for g in groups]

    # Master lits
    s_all = set(chain(*groups))

    # bin(3) --> '0b11', so bin(3).split('0b')[-1] will remove "0b"
    set_collections = {}
    for ii in range(1, 2 ** n):
        key = bin(ii).split('0b')[-1].zfill(n)
        value = s_all
        sets_for_intersection = [sets_data[i] for i in range(n) if key[i] == '1']
        sets_for_difference = [sets_data[i] for i in range(n) if key[i] == '0']
        for s in sets_for_intersection:
            value = value & s
        for s in sets_for_difference:
            value = value - s
        dict_key = '∩'.join([labels[i] for i in range(n) if key[i] == '1'])
        set_collections[dict_key] = value

    set_sizes = {k: len(v) for k, v in set_collections.items()}
    set_sizes = pd.DataFrame(pd.Series(set_sizes), columns=['size'])

    return set_sizes, set_collections