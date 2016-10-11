import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = 'Arial'
from palettable.colorbrewer.qualitative import Dark2_8


def volcano_plot(df, p_value=0.05, log2_fc=1, x_colname='logFC', y_colname='adj.P.Val', cutoff_lines=True, top_n=None,
                 top_by='adj.P.Val'):

    # Keep NaNs for reporting, split dataframe into two dataframes based on cutoffs
    df['-log10(p)'] = -np.log10(df[y_colname])
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

    fig, ax = plt.subplots(figsize=(10, 10))
    # Split top data points if requested
    if top_n:
        ascending = True if top_by == 'adj.P.Val' else False
        sig.sort_values(top_by, ascending=ascending)
        top_sig = sig[:top_n]
        sig = sig[top_n:]
        ax.plot(top_sig[x_colname], top_sig['-log10(p)'], 'o', c=Dark2_8.mpl_colors[0], ms=10, zorder=2)
        for row in top_sig.iterrows():
            plt.annotate(row[0], xy=(row[1]['logFC'], row[1]['-log10(p)']), fontsize=16, style='italic')
    # Make plot
    ax.plot(sig[x_colname], sig['-log10(p)'], 'o', c=Dark2_8.mpl_colors[2], ms=10, zorder=1)
    ax.plot(insig[x_colname], insig['-log10(p)'], 'o', c=Dark2_8.mpl_colors[-1], ms=10, zorder=0, mew=0)

    # Adjust axes
    ax.set_xlim([-max_x, max_x])
    ax.set_ylim([0, max_y])

    # Add cutoff lines
    if cutoff_lines:
        color = Dark2_8.mpl_colors[1]
        ax.plot([-max_x, max_x], [-np.log10(p_value), -np.log10(p_value)], '--', c=color, lw=3)
        ax.plot([-log2_fc, -log2_fc], [0, max_y], '--', c=color, lw=3)
        ax.plot([log2_fc, log2_fc], [0, max_y], '--', c=color, lw=3)

    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlabel(r'$log_2(\frac{KO}{WT})$', fontsize=28)
    ax.set_ylabel(r'$-log_{10}$(corrected p-value)', fontsize=28)
    plt.show()
