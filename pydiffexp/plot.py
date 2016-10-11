import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = 'Arial'


def volcano_plot(df, p_value=0.05, log2_fc=1, x_colname='logFC', y_colname='adj.P.Val',
                 cutoff_lines=True):

    df['-log10(p)'] = -np.log10(df[y_colname])

    # Keep NaNs for reporting, split dataframe into two dataframes based on cutoffs
    nans = df[df.isnull().any(axis=1)]
    df = df.dropna()
    sig = df[(df[y_colname] <= p_value) & (np.abs(df[x_colname]) >= log2_fc)]
    insig = df[~(df[y_colname] <= p_value) | ~(np.abs(df[x_colname]) >= log2_fc)]

    max_y = np.max(sig['-log10(p)'])
    max_x = np.ceil(np.max(np.abs(sig[x_colname])))

    # Change number of ticks if necessary
    if max_x <= 10:
        rounded_lim = int(2 * np.floor(max_x/2))
        xticks = np.arange(-rounded_lim, rounded_lim+2, 2)

    # Make plot

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(sig[x_colname], sig['-log10(p)'], marker='o', c='b', ls='', ms=10, markerfacecoloralt='green', fillstyle='right')
    print(insig.idxmax())
    sys.exit()
    ax.plot(insig[x_colname], insig['-log10(p)'], marker='o', c='k', ls='None', ms=10)
    plt.show()
    # ax.scatter(insig[x_colname], insig['-log10(p)'], c='k', s=100)
    ax.set_xlim([-max_x, max_x])
    ax.set_ylim([0, max_y])

    if cutoff_lines:
        ax.plot([-log2_fc, -log2_fc], [0, max_y], '--', c='r', lw=3)
        ax.plot([log2_fc, log2_fc], [0, max_y], '--', c='r', lw=3)
        ax.plot([-max_x, max_x], [-np.log10(p_value), -np.log10(p_value)], '--', c='r', lw=3)

    # for ii, xy in enumerate(zip(log2_fc[top_idx], adj_pval[top_idx])):
       # plt.annotate(results_df.iloc[ii,0].capitalize(), xy=xy, fontsize=16, style='italic')

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlabel(r'$log_2(\frac{KO}{WT})$', fontsize=28)
    ax.set_ylabel(r'$-log_{10}(pval)$', fontsize=28)
    plt.tight_layout()
    plt.show()
