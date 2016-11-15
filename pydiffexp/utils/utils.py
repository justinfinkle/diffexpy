def grepl(search_list, substr):
    grep_list = list(filter(lambda x: substr in x, search_list))
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
