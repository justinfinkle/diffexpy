import sys, inspect
import pandas as pd
import numpy as np
from scipy import stats
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, fill_between
import matplotlib as mpl

import palettable.colorbrewer as cbrewer

# Set plot defaults
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = 'Arial'
axes = {'labelsize': 28,
        'titlesize': 28}
mpl.rc('axes', **axes)
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 24

_colors = cbrewer.qualitative.Dark2_8.mpl_colors
_paired = cbrewer.qualitative.Paired_9.mpl_colors


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
            fs = mpl.rcParams['legend.fontsize']
            for row in top_sig.iterrows():
                ax.annotate(row[0], xy=(row[1][x_colname], row[1][y_colname]), fontsize=fs, style='italic')

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

    ax.legend(loc='best', numpoints=1)

    # Adjust labels
    ax.tick_params(axis='both', which='major')
    ax.set_xlabel(r'$log_2(\frac{KO}{WT})$')
    ax.set_ylabel(r'$-log_{10}$(corrected p-value)')
    plt.show()


def add_ts(ax, data, name, subgroup='time', mean_line_dict=None, fill_dict=None):
    gene = data.name
    data = data.reset_index()
    grouped_data = data.groupby(subgroup)
    grouped_stats = np.array([[g, np.mean(data[gene]), stats.sem(data[gene])] for g, data in grouped_data]).T
    if mean_line_dict is None:
        mean_line_dict = dict()
    if fill_dict is None:
        fill_dict = dict()
    mean_defaults = dict(ls='-', marker='s', lw=2, mew=0, label=(name+" mean"), ms=10, zorder=0)
    mean_kwargs = dict(mean_defaults, **mean_line_dict)
    mean_line, = ax.plot(grouped_stats[0], grouped_stats[1], **mean_kwargs)
    mean_color = mean_line.get_color()
    jitter_x = data[subgroup]#+(np.random.normal(0, 1, len(data)))
    ax.plot(jitter_x, data[gene], '.', color=mean_color, ms=15, label='', alpha=0.5)

    fill_defaults = dict(lw=0, facecolor=mean_color, alpha=0.2, label="")
    fill_kwargs = dict(fill_defaults, **fill_dict)
    ax.fill_between(grouped_stats[0], grouped_stats[1] - grouped_stats[2], grouped_stats[1] + grouped_stats[2],
                    **fill_kwargs)


def tsplot(df, supergroup='condition', subgroup='time'):
    gene = df.name
    supers = set(df.index.get_level_values(supergroup))
    fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', _colors))
    for sup in supers:
        sup_data = df.loc[sup]
        add_ts(ax, sup_data, sup, subgroup=subgroup)
    # ax.set_xlim([np.min(grouped_stats[0]), np.max(grouped_stats[0])])
    ax.legend(loc='best', numpoints=1)
    # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: '{:,.0f}'.format(x)))
    ax.set_xlabel(subgroup.title())
    ax.set_ylabel('Expression')
    ax.set_title(gene)
    plt.tight_layout()
    plt.show()
