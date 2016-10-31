def grepl(search_list, substr):
    grep_list = list(filter(lambda x: substr in x, search_list))
    return grep_list


def filter_value(value, axis=0, criteria='any'):
    pass