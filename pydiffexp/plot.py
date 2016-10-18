import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable.colorbrewer as cbrewer

# Set plot defaults
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = 'Arial'
_colors = cbrewer.qualitative.Dark2_8.mpl_colors


def volcano_plot(df: pd.DataFrame, p_value: float=0.05, fc=2, x_colname='logFC', y_colname='-log10p',
                 cutoff_lines=True, top_n=None, top_by='-log10p', show_labels=False):

    # Get rid of NaN data
    df = df.dropna()

    # Convert cutoffs to logspace
    log2_fc = np.log2(fc)
    log10_pval = -np.log10(p_value)

    # Split data into above and below cutoff dataframes
    sig = df[(df[y_colname] >= log10_pval) & (np.abs(df[x_colname]) >= log2_fc)]
    insig = df[~(df[y_colname] >= log10_pval) | ~(np.abs(df[x_colname]) >= log2_fc)]

    # Get maximum values for formatting latter
    max_y = np.max(sig[y_colname])
    max_x = np.ceil(np.max(np.abs(sig[x_colname])))

    fig, ax = plt.subplots(figsize=(10, 10))

    # Split top data points if requested
    if top_n:

        # Find points to highlight
        sort = set()
        if isinstance(top_by, list):
            for col in top_by:
                sort = sort.union(set(sig.index[np.argsort(np.abs(sig[col]))[::-1]][:top_n].values))
        elif isinstance(top_by, str):
            sort = sort.union(set(sig.index[np.argsort(np.abs(sig[top_by]))[::-1]][:top_n].values))
        else:
            raise ValueError('top_by must be a string or list of values found in the DataFrame used for the plot')

        top_sig = sig.loc[sort]
        sig = sig.drop(sort)
        ax.plot(top_sig[x_colname], top_sig[y_colname], 'o', c=_colors[0], ms=10, zorder=2, label='Top Genes')

        if show_labels:
            for row in top_sig.iterrows():
                plt.annotate(row[0], xy=(row[1]['logFC'], row[1][y_colname]), fontsize=24, style='italic')

    # Make plot
    ax.plot(sig[x_colname], sig[y_colname], 'o', c=_colors[2], ms=10, zorder=1, label='Diff Exp')
    ax.plot(insig[x_colname], insig[y_colname], 'o', c=_colors[-1], ms=10, zorder=0, mew=0, label='')

    # Adjust axes
    ax.set_xlim([-max_x, max_x])
    ax.set_ylim([0, max_y])

    # Add cutoff lines
    if cutoff_lines:
        color = _colors[1]
        # P value line
        ax.plot([-max_x, max_x], [log10_pval, log10_pval], '--', c=color, lw=3, label='Threshold')

        # log fold change lines
        ax.plot([-log2_fc, -log2_fc], [0, max_y], '--', c=color, lw=3)
        ax.plot([log2_fc, log2_fc], [0, max_y], '--', c=color, lw=3)

    ax.legend(loc='best', numpoints=1, fontsize=24)

    # Adjust labels
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlabel(r'$log_2(\frac{KO}{WT})$', fontsize=28)
    ax.set_ylabel(r'$-log_{10}$(corrected p-value)', fontsize=28)
    plt.show()


def tsplot():
    pass
