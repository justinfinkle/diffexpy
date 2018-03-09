import sys
import warnings

import numpy as np
import pandas as pd


def parse_title(title, split_str=" "):
    """
    Parse the title of GSE13100 into usable metadata. Should work with pandas apply()
    Args:
        title:
        split_str:

    Returns:

    """
    split = title.split(split_str)
    meta = []
    if len(split) == 2:
        meta = [split[0], "NA", "NA", split[1]]
    elif len(split) == 8:
        meta = ['MUT', split[4], int(split[5].replace('t', "")), split[-1]]
    elif len(split) == 6:
        meta = ['WT', split[2], int(split[3].replace('t', "")), split[-1]]
    return pd.Series(meta, index=['condition', 'rna_type', 'time', 'rep'])


def mi_to_array(mi):
    labels = np.array(mRNA.columns.labels)
    x = np.array([lev[labels[ii]].values.tolist() for ii, lev in enumerate(mi.levels)])
    return x.T


if __name__ == '__main__':
    # Change output for easier reading
    pd.set_option('display.width', 200)

    # Parse R GEOQuery data into a pandas multiindex
    data = pd.read_csv("../data/GSE13100/GSE13100_BgCorrected_data.csv", index_col=0)
    row_info = pd.read_csv("../data/GSE13100/GSE13100_BgCorrected_rowinfo.csv").fillna("NA")
    col_info = pd.read_csv("../data/GSE13100/GSE13100_BgCorrected_colinfo.csv")

    # Compile the experiment information
    exp_info = col_info.title.apply(parse_title)
    exp_info.insert(0, 'geo', col_info.geo_accession.values)

    # Make sure the order matches the data
    data.sort_index(axis=1, ascending=True, inplace=True)
    exp_info.sort_values('geo', ascending=True, inplace=True)
    if not np.array_equal(data.columns.values, exp_info.geo.values):
        warnings.warn('Data columns and experimental info are not equal. Values may not match labels')

    # Make the columns a multiindex
    data.columns = pd.MultiIndex.from_arrays(exp_info.values.T.tolist(), names=exp_info.columns.values)
    data.sort_index(axis=1, inplace=True)

    # Select only the RA data and quantile normalize it. Log2 tranform will happen in pydiffexp pipeline
    idx = pd.IndexSlice
    mRNA = data.loc[:, idx[:, :, 'TR', :, :]]
    mRNA.to_pickle("../data/GSE13100/bgcorrected_GSE13100_TR_data.pkl")
    sys.exit()

    # Plot the distributions of the RA abundance
    info_idx = mi_to_array(mRNA.columns)
    fig, ax = plt.subplots(6, 7)
    ax_list = ax.flatten()
    for ii in range(mRNA.shape[1]):
        color = 'red' if info_idx[ii, 1] == 'MUT' else 'blue'
        title = 'time: {}, rep: {}'.format(info_idx[ii, 3], info_idx[ii, 4])
        ax_list[ii].hist(np.log2(mRNA.values[:, ii]+1), color=color)
        ax_list[ii].set_title(title)
    plt.show()

    # Pickle Data
    mRNA.to_pickle('../data/GSE13100/log2_bgcorrected_GSE13100_TR_data.pkl')

