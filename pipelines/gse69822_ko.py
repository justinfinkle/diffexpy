from collections import Counter
from collections import OrderedDict
from typing import Union

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.transforms import Affine2D
from palettable.cartocolors.qualitative import Bold_8, Prism_10
from scipy import integrate, stats
from sklearn.utils import shuffle

from pydiffexp import DEPlot
from pydiffexp.pipeline import DynamicDifferentialExpression as DDE
from pydiffexp.plot import elbow_criteria
from pydiffexp.utils import multiindex_helpers as mi

# Set defaults
mpl.rcParams['axes.labelweight'] = 'bold'


def load_data(path, mi_level_names, bg_shift=True, **kwargs):
    """

    :param path:
    :param bg_shift:
    :param kwargs:
    :return:
    """
    kwargs.setdefault('sep', ',')
    kwargs.setdefault('index_col', 0)

    # Load original data
    raw_data = pd.read_csv(path, **kwargs)

    if bg_shift:
        raw_data[raw_data <= 0] = 1

    df = mi.make_multiindex(raw_data, index_names=mi_level_names)

    return df


def running_stat(x, N, s='median'):
    """

    :param x:
    :param N:
    :param s: running statistic to calculate
    :return:
    """
    if s == 'median':
        rs = np.array([np.median(x[ii:ii+N]) for ii in range(len(x)-N+1)])
    elif s == 'mean':
        rs = np.cumsum(np.insert(x, 0, 0))
        rs = (rs[N:] - rs[:-N]) / float(N)

    return rs


def plot_gene_prediction(gene, match, data, sim_der, col_names, ensembl_to_hgnc, ax=None, **kwargs):
    dep = DEPlot()
    matching_sim = match.loc[match.train_gene==gene, 'net'].astype(str).values
    pred_lfc = sim_der.coefficients.loc[matching_sim]
    baseline = data.loc[gene, 'wt'].groupby('time').mean().values
    random = sim_der.coefficients+baseline
    random.columns=col_names
    random = random.unstack()
    random.index = random.index.set_names(['time', 'replicate'])
    random.name='random'
    random = pd.concat([random], keys=['random'], names=['condition'])
    pred = pred_lfc+baseline
    pred.columns = col_names
    pred = pred.unstack()
    pred.index = pred.index.set_names(['time', 'replicate'])
    pred.name = 'predicted'
    pred = pd.concat([pred], keys=['predicted'], names=['condition'])
    true = data.loc[gene, ['ki']]
    pred = pred.reorder_levels(true.index.names)
    random = random.reorder_levels(true.index.names)
    ts_data = pd.concat([true, pred, random])
    ts_data.name = gene
    ax = dep.tsplot(ts_data, scatter=False, ax=ax, legend=False, **kwargs)
    ax.set_title(ensembl_to_hgnc.loc[gene, 'hgnc_symbol'])
    ax.set_ylabel('')


def regular_points_on_circle(startangle=30, points=3, rad=1):
    sa_rad = startangle*np.pi/180
    rand_angles = np.linspace(0+sa_rad, np.pi*2+sa_rad, num=points+1)[:-1]
    x = np.cos(rand_angles)*rad
    y = np.sin(rand_angles)*rad
    return np.vstack([x,y]).T


def plot_net(ax, node_info, models, labels, cmap, pie_colors):
    pal = cmap
    pie_centers = {labels[n]: point for n, point in enumerate(regular_points_on_circle())}
    pie_rad = 0.3
    # pie_colors = [Prism_10.mpl_colors[idx] for idx in [3, 5, 7]]+['0.5']
    # pie_colors = [Tropic_7.mpl_colors[0], Tropic_7.mpl_colors[-1], '0.2', '0.7']

    for node, center in pie_centers.items():
        ax.annotate(xy=center, s=node, color='w', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.pie(node_info[node]['fracs'], startangle=90, radius=pie_rad,
               center=center, colors=pie_colors, wedgeprops={'linewidth':0})

    center_lines = {}
    for combo in [('x', 'G'), ('G', 'y'), ('y', 'x')]:
        center_lines[combo] = np.array([pie_centers[combo[0]], pie_centers[combo[1]]])

    for key, val in center_lines.items():
        start, end = val[0], val[1]
        o_length = np.sqrt(np.sum((start-end)**2))
        center_xy = (end-start)/2+start

        # Shorten
        ss, se = Affine2D().scale((o_length-2*pie_rad)/o_length).transform([start, end])

        # Translate by calculating where the center should have been
        new_center_xy = (se-ss)/2+ss
        center_delta = (new_center_xy-center_xy)
        ss -= center_delta
        se -= center_delta
        new_center_xy = (se-ss)/2+ss

        # Translate outward, "left"
        parent, child = key
        edge = '{}->{}'.format(parent, child)
        edge_count = models[edge].abs().sum()
        edge_sign = models[edge].sum()/edge_count
        edge_frac = edge_count/len(models)
        ec = pal(edge_sign+1/2)

        dx, dy = (se-ss)/2
        fraction = 1/6
        translate = np.array([-dy, dx])*fraction
        left_center = center_xy-translate

        center_delta = (new_center_xy-left_center)
        out_s = ss-center_delta
        out_e = se-center_delta

        hw = 12
        hl = 0.7*hw
        tw = 0.7*hw*edge_frac
        astyle = ArrowStyle('simple', head_length=hl, head_width=hw, tail_width=tw)
        fa = FancyArrowPatch(posA=out_s, posB=out_e, arrowstyle=astyle, lw=0, color=ec)
        ax.add_artist(fa)

        # Translate inward, "right"
        parent, child = key[1], key[0]
        edge = '{}->{}'.format(parent, child)
        edge_count = models[edge].abs().sum()
        edge_sign = models[edge].sum()/edge_count
        edge_frac = edge_count/len(models)
        ec = pal(edge_sign+1/2)

        right_center = center_xy+translate

        center_delta = (new_center_xy-right_center)
        # Flip direction
        in_s = se-center_delta
        in_e = ss-center_delta

        tw = 0.7 * hw * edge_frac
        astyle = ArrowStyle('simple', head_length=hl, head_width=hw, tail_width=tw)
        fa = FancyArrowPatch(posA=in_s, posB=in_e, arrowstyle=astyle, lw=0, color=ec)
        ax.add_artist(fa)

    ax.axis('equal')
    ax.set_ylabel("n = {}".format(len(models)), rotation=0, va='top', fontweight='normal')


def load_sim_data(path, node='y', perturb: Union[int, tuple, None]=1):
    """
    Load simulation data
    :param path:
    :param node:
    :param perturb:
    :return:
    """
    # Initizialize a slicer
    idx = pd.IndexSlice

    # Read data
    sim_data = pd.read_pickle(path)

    if perturb is not None:
        sim_data = sim_data.loc[node, idx[:, :, perturb, :]]
    sim_data.columns = sim_data.columns.remove_unused_levels()
    sim_data.columns.set_names(['replicate', 'time'], level=[1, 3], inplace=True)
    return sim_data


def plot_error_predictor(x, y, c='KI', out_path=None, ax=None):
    print(stats.spearmanr(x, y))
    with sns.axes_style("whitegrid"):
        ax = sns.regplot(x, y, ax=ax)
    ax.set_xlabel('log2(Mean {}-WT LFC)'.format(c))
    ax.set_ylabel('∆MSE')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, fmt='pdf')
        plt.close()
    else:
        plt.show()


def censor_values(df, col, thresh=3):
    """

    :param df:
    :param col:
    :param thresh: float; multiplier for the stddev
    :return:
    """
    df = df[df[col] < (thresh*df[col].std())]
    return df


def calc_groups(df):
    df = df.sort_values('mean_abs_lfc', ascending=False)
    censored = censor_values(df, 'group_dev')
    n_top, _ = elbow_criteria(range(len(censored)), censored.mean_abs_lfc)
    test_sort = censored.sort_values('abs_dev', ascending=False)
    return df, censored, n_top, test_sort


def plot_top_cut(df, n_top, out_path=None):
    with sns.axes_style('whitegrid'):
        plt.plot(range(len(df)), df.mean_abs_lfc, label='sorted mean dev', lw=3,
                 c='k')
        max_val = df.mean_abs_lfc.max()
        plt.plot([n_top, n_top], [0, max_val], '--', label='cutoff')
        plt.xlim(0, len(df))
        plt.ylim(0, max_val)
        plt.xlabel('Ranked Gene Prediction')
        plt.ylabel(r'Mean Absolute LFC($\frac{KO}{WT}$)')
        plt.tight_layout()
    if out_path:
        plt.savefig(out_path, fmt='pdf')
        plt.close()
    else:
        plt.show()


def plot_running_stat(unsorted, censored, ts, top, n_shuff=100, out_path=None):
    n_top = len(top)
    rm = np.array([running_stat(shuffle(unsorted.percent.values), n_top, s='median') for ii in range(n_shuff)])
    plt.plot(rm.T, c='0.5', zorder=0, alpha=0.1)
    plt.plot([0, 0], lw=2, c='0.5', label='random_sort', zorder=0)
    plt.plot(np.median(rm, axis=0), label='<random_sort>')
    srm = running_stat(ts.percent.values, n_top, s='median')
    plt.plot([0, len(ts)], [ts.percent.median(), ts.percent.median()], label='med(all models)')
    plt.plot([0, len(ts)], [top.percent.median(), top.percent.median()], label='med(top models)')
    plt.plot(srm, label='LFC sorted')
    plt.plot([0, len(ts)], [0, 0], 'k-', lw=1)
    plt.ylim(-100, 100)
    plt.plot(range(len(censored)), censored.mean_abs_lfc * 10, label='mean_abs_lfc')
    plt.xlim([0, len(srm) - 1])
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, fmt='pdf')
        plt.close()
    else:
        plt.show()


def calc_pr(x, axis=1):
    """

    :param x: matrix of 1s and 0s
    :param axis:
    :return:
    """

    # Total number of classifications made
    n_samp = (np.arange(x.shape[axis]) + 1)

    if axis == 1:
        x = x.T
        axis = 0
        n_samp = n_samp[:, None]
    # Calculate true positives as each new value is added
    cum_tp = np.cumsum(x, axis=axis)
    cum_fp = np.cumsum(~x, axis=axis)

    # Total number of true positives
    total_tp = np.sum(x, axis=axis)
    total_fp = np.sum(~x, axis=axis)

    # True positive rate
    tpr = cum_tp/total_tp

    # False positive rate
    fpr = cum_fp/total_fp

    # Precision
    p = cum_tp/n_samp

    return tpr, fpr, p


def random_order(x, n_shuff=100):
    shuffled = np.array([shuffle(x) for _ in range(n_shuff)])
    return shuffled


def plot_pr(censored, test_sort, n_top, out_path):
    with sns.axes_style('whitegrid'):
        plt.figure(figsize=(10, 10))
        shuffled = random_order(censored.percent.values)
        correct_class = shuffled > 0
        s_recall, s_fpr, s_precision, = calc_pr(correct_class)
        plt.plot(s_recall, s_precision, '0.5', alpha=0.5)
        plt.plot(np.mean(s_recall, axis=1), np.mean(s_precision, axis=1), 'k')

        censored_correct = censored.percent > 0
        c_recall, c_fpr, c_precision, = calc_pr(censored_correct, axis=0)
        c_aupr = integrate.cumtrapz(c_precision, c_recall)[-1]
        print('KO LFC AUPR: ', c_aupr)
        c_label = 'Train LFC ({:0.2f})'.format(c_aupr)
        plt.plot(c_recall, c_precision, lw=3, label=c_label, color=Prism_10.mpl_colors[1])

        ss_correct = test_sort.percent > 0
        ss_recall, ss_fpr, ss_precision, = calc_pr(ss_correct, axis=0)
        ss_aupr = integrate.cumtrapz(ss_precision, ss_recall)[-1]
        print('KI LFC AUPR: ', ss_aupr)
        ss_label = 'Train LFC ({:0.2f})'.format(ss_aupr)
        plt.plot(ss_recall, ss_precision, lw=3, label=ss_label, color=Prism_10.mpl_colors[-4])

        plt.plot([n_top / len(censored), n_top / len(censored)], [0, 1], '--', lw=3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(loc='best', handlelength=1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.tight_layout()
    if out_path:
        plt.savefig(out_path, fmt='pdf')
        plt.close()
    else:
        plt.show()


def center_axis(axes: plt.Axes, which='y'):
    if which == 'y':
        max_abs = np.max(np.abs(axes.get_ylim()))
        axes.set_ylim(-max_abs, max_abs)
    elif which == 'x':
        max_abs = np.max(np.abs(axes.get_xlim()))
        axes.set_xlim(-max_abs, max_abs)
    elif which == 'both':
        pass
    return


def box_plots(ts, top, spec, top_color):
    group_keys = ['full', 'top']
    group_colors = ['0.5'] + [top_color]
    color_dict = OrderedDict(zip(group_keys, group_colors))
    box_data = [ts.percent, top.percent]
    df = pd.concat(box_data, keys=color_dict.keys()).reset_index()
    df.columns = ['dataset', 'gene', 'value']
    df['metric'] = 'percent'

    box_ax1 = plt.subplot(spec[1, 0])
    box_ax2 = plt.subplot(spec[1, 1])

    box_ax1 = sns.boxplot(data=df, x='metric', y='value', hue='dataset', showfliers=False, width=0.5, notch=True,
                         medianprops=dict(solid_capstyle='butt', color='k', lw=2), palette=color_dict,
                         boxprops=dict(linewidth=0), ax=box_ax1)
    box_ax1.plot(box_ax1.get_xlim(), [0, 0], 'k-', zorder=0)
    box_ax1.set_ylabel('∆MSE (%)')
    box_ax1.set_xticklabels([])
    box_ax1.set_xlabel('')
    box_ax1.legend().remove()
    center_axis(box_ax1)

    violin_data = [ts.grouped_diff, top.grouped_diff]
    violin_df = pd.concat(violin_data, keys=color_dict.keys()).reset_index()
    violin_df.columns = ['dataset', 'gene', 'value']
    violin_df['metric'] = 'difference'
    box_ax2 = sns.boxplot(data=violin_df, x='metric', y='value', hue='dataset', showfliers=False, width=0.5, notch=True,
                         medianprops=dict(solid_capstyle='butt', color='k', lw=2), palette=color_dict,
                         boxprops=dict(linewidth=0), ax=box_ax2)

    box_ax2.plot(box_ax2.get_xlim(), [0, 0], 'k-', zorder=0)
    box_ax2.yaxis.set_label_position('right')
    box_ax2.yaxis.set_ticks_position('right')
    box_ax2.set_ylabel('∆MSE', rotation=270, va='bottom')
    box_ax2.set_xticklabels([])
    box_ax2.set_xlabel('')
    box_ax2.set_ylim(box_ax2.get_ylim())
    leg = box_ax2.legend(loc='upper center', ncol=2, bbox_to_anchor=[0, 0.05], frameon=False, handlelength=1.5,
                   handletextpad=0.5, handleheight=0.5)
    for patch in leg.get_patches():
        patch.set_linewidth(0)
    center_axis(box_ax2)

    # Set the line widths just for the outer lines on the violinplots
    lw = 0
    from matplotlib.collections import PolyCollection

    for art in box_ax2.get_children():
        if isinstance(art, PolyCollection):
            art.set_linewidth(lw)

    return


def auroc_plots(ax, censored, test_sort, n_top):
    shuffled = random_order(censored.percent.values)
    censored_correct = censored.percent > 0
    test_correct = test_sort.percent > 0
    shuffled_correct = shuffled > 0
    c_tpr, c_fpr, _ = calc_pr(censored_correct, axis=0)
    t_tpr, t_fpr, _ = calc_pr(test_correct, axis=0)
    s_tpr, s_fpr, _ = calc_pr(shuffled_correct, axis=1)

    # Plot shuffles
    ax.plot(s_fpr, s_tpr, c='0.5', alpha=0.1)
    ax.plot([0, 0], [0, 0], lw=2, c='0.5', label='Shuffled')

    # Random line
    ax.plot([0, 1], [0, 1], 'k')

    # Auroc lines
    ax.plot(c_fpr, c_tpr, lw=3, label='Train LFC', color=Prism_10.mpl_colors[1])
    ax.plot(t_fpr, t_tpr, lw=3, label='Test LFC', color=Prism_10.mpl_colors[-4])

    # Top line
    fpr_cut = (np.cumsum(censored.percent < 0) / np.sum(censored.percent < 0)).iloc[n_top]
    top_line, = ax.plot([fpr_cut, fpr_cut], [0, 1], '--', lw=3, color=Prism_10.mpl_colors[-2])

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([0, 0.5, 1])
    tpr = sum(censored_correct)/len(censored)
    fpr = 1-tpr
    ax.set_ylabel('Normalized TPR ({:.2f})'.format(tpr))
    ax.set_xlabel('Normalized FPR ({:.2f})'.format(fpr))

    leg = ax.legend(handlelength=1, loc='center left', bbox_to_anchor=(0.97, 0.5),
                    handletextpad=0.2, frameon=False)

    for line in leg.get_lines():
        line.set_lw(4)
        line.set_solid_capstyle('butt')

    return top_line.get_color()


def despine(ax, spines=('top', 'bottom', 'left', 'right')):
    for spine in spines:
        ax.spines[spine].set_visible(False)


def detick(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def stars(p):
    if p < 0.0001:
        s = "****"
    elif p < 0.001:
        s = "***"
    elif p < 0.01:
        s = "**"
    elif p < 0.05:
        s = "*"
    else:
        s = ""
    return s


def plot_genes(sub_spec, leg_spec, net_spec, matches, dde, sim_dea, tx_to_gene, net_data, ts):
    # genes = [top.index[29],  ts.sort_values('percent', ascending=False).index[0],
    #          ts.sort_values('percent', ascending=False).index[30]]
    # genes = ts.index[np.random.randint(0, len(top), size=3)]
    # genes = top.sort_values('grouped_e').index[:3]
    # genes = ['ENSG00000117289', 'ENSG00000213626', 'ENSG00000170044']
    genes = ['ENSG00000074054', 'ENSG00000102760', 'ENSG00000131759']

    w_ratios = [1]*len(genes)+[0.1, 0.1]
    cols = len(genes)+2
    gs_top = gridspec.GridSpecFromSubplotSpec(2, cols, subplot_spec=sub_spec,
                                              width_ratios=w_ratios, wspace=0.5)

    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=leg_spec,
                                                 width_ratios=[1, 1, 0.1, 0.1],
                                                 wspace=0.5)
    gs_mid = gridspec.GridSpecFromSubplotSpec(1, cols, subplot_spec=net_spec,
                                              width_ratios=w_ratios, wspace=0.5)

    train_conditions = list(dde.training.values())
    dep = DEPlot()
    greys = ['0.1', '0.7', '0.4', '0.1', '0.7']
    conditions = ['wt', 'ko', 'ki', 'predicted', 'random']
    colors = {c: idx for c, idx in zip(conditions, greys)}
    # colors['random'] = '0.7'
    three_col = ['#AA0000', '#BBBBBB', '#0000AA']
    new_map = LinearSegmentedColormap.from_list('custom', three_col)
    edge_cmap = new_map
    pie_colors = ['#0F8554', '#E17C05', '0.2', '0.7']
    train_markers = ['o', '^']
    test_markers = ['d', 'X', 's']
    test_stats = ['mean', 'median', 'median']
    test_lines = ['-', '-', '--']
    plot_times = dde.times.copy()
    plot_times.remove(15)
    for idx, gene in enumerate(genes):
        with sns.axes_style("whitegrid"):
            train_ax = plt.subplot(gs_top[0, idx])
            pred_ax = plt.subplot(gs_top[1, idx])
            dep.tsplot(dde.dea.data.loc[gene, train_conditions], ax=train_ax, legend=False,
                       no_fill_legend=True, color_dict=colors, scatter=False, markers=train_markers)
            train_ax.set_title(tx_to_gene.loc[gene, 'hgnc_symbol'])
            train_ax.set_ylabel('log2(counts)')
            train_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            train_ax.set_xlabel('')
            train_ax.set_xlim(min(plot_times), max(plot_times))
            train_ax.set_xticks(plot_times)
            train_ax.set_xticklabels([])

            plot_gene_prediction(gene, matches, dde.dea.data, sim_dea.results['ki-wt'], sim_dea.times, tx_to_gene,
                                 ax=pred_ax, no_fill_legend=True, color_dict=colors, markers=test_markers,
                                 stats=test_stats, ls=test_lines)
            pred_ax.set_xlim(min(plot_times), max(plot_times))
            pred_ax.set_xticks(plot_times)
            pred_ax.set_xticklabels(plot_times, rotation=90)
            pred_ax.set_ylabel('log2(counts)')
            pred_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            pred_ax.set_title('')

            if idx != 0:
                train_ax.set_ylabel('')
                pred_ax.set_ylabel('')

        net_ax = plt.subplot(gs_mid[0, idx])
        labels = ['G', 'x', 'y']
        logics = ['_multiplicative', '_linear', '']
        node_info = {}
        networks = matches[matches.train_gene == gene]['net'].values
        networks = [int(net.replace('-', '')) for net in networks]
        models = net_data.loc[networks]
        for node in labels:
            cur_dict = {}
            counts = Counter(models['{}_logic'.format(node)])
            cur_dict['fracs'] = [counts['{}{}'.format(node, log)] for log in logics]
            no_in = sum(models['{}_in'.format(node)] == 0)
            cur_dict['fracs'][-1] -= no_in
            cur_dict['fracs'].append(no_in)
            node_info[node] = cur_dict
        plot_net(net_ax, node_info, models, labels, edge_cmap, pie_colors)

        # Add the net legend
        if idx == len(genes)-1:
            leg = net_ax.legend(['AND', 'OR', 'Single input', 'No input'], loc='center left',
                                bbox_to_anchor=(1, 0.5), handletextpad=0.5, frameon=False,
                                handlelength=1)
            leg.set_title("Node \nRegulation", prop={'size': 28, 'weight': 'bold'})

    # Add the legends
    labels = {'wt': 'Wildtype', 'ko': 'PI3K KO', 'ki': 'PI3K KI'}
    for line in train_ax.get_lines()+pred_ax.get_lines():
        label = line.get_label()
        try:
            new = labels[label]
        except KeyError:
            new = label.capitalize()
        line.set_label(new)

    train_leg = train_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
                                handlelength=1)
    pred_leg = pred_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False,
                              handlelength=1)

    train_leg.set_title('Training', prop={'size': 28, 'weight': 'bold'})
    pred_leg.set_title('Testing', prop={'size': 28, 'weight': 'bold'})

    # Add arrow
    sns.set_style(None)
    arrow_ax = plt.subplot(gs_bottom[0])
    despine(arrow_ax)
    detick(arrow_ax)

    # Scale arrow width
    fig = arrow_ax.get_figure()
    bbox = arrow_ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    height *= fig.dpi/2
    astyle = ArrowStyle('wedge', tail_width=height, shrink_factor=0.5)
    fa = FancyArrowPatch(posA=[0, 0.5], posB=[1, 0.5], arrowstyle=astyle, lw=0, color='k')
    arrow_ax.add_artist(fa)
    arrow_ax.set_title('Fraction of models \n edge exists')
    arrow_ax.set_xticks([0, 1])
    arrow_ax.set_xticklabels(['100%', '0%'])

    # Add colorbar
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    cbar_ax = plt.subplot(gs_bottom[1])
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=edge_cmap,
                                    norm=norm, orientation='horizontal')
    cb1.set_ticks([-1, 0, 1])
    cbar_ax.set_title('Average edge sign')


def panel_plot(ts, top, censored, test_sort, matches, dde, tx_to_gene, sim_dea, net_data):
    n_top = len(top)
    ts_percent_p = stats.wilcoxon(ts.percent).pvalue / 2
    top_percent_p = stats.wilcoxon(top.percent).pvalue / 2
    ts_delta_p = stats.wilcoxon(ts.grouped_diff).pvalue / 2
    top_delta_p = stats.wilcoxon(top.grouped_diff).pvalue / 2
    print("All % median: {}, % Top median: {}".format(ts.percent.median(), top.percent.median()))
    print()
    print("All % wilcoxp: {}, % Top wilcoxp: {}".format(ts_percent_p,
                                                        top_percent_p))
    print()
    print("All ∆ median: {}, ∆ Top median: {}".format(ts.grouped_diff.median(), top.grouped_diff.median()))
    print()
    print("All ∆ wilcoxp: {}, ∆ Top wilcoxp: {}".format(ts_delta_p, top_delta_p))

    plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 0.5, 0.1],
                           wspace=0.3, hspace=0.6)

    # Create a gridspec within the gridspec. 1 row and 2 columns, specifying width ratio
    gs_right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 1], hspace=0.3)

    auroc_ax = plt.subplot(gs_right[0, :])
    box_spec = gs_right

    # AUROC
    top_color = auroc_plots(auroc_ax, censored, test_sort, n_top)

    # Make boxplots
    box_plots(ts, top, box_spec, top_color)

    # Plot gene examples
    left_top = gs[0, 0]
    left_mid = gs[1, 0]
    left_bottom = gs[2, 0]
    plot_genes(left_top, left_bottom, left_mid, matches, dde, sim_dea, tx_to_gene, net_data, ts)

    # Adjust axes
    plt.subplots_adjust(left=0.055, right=0.90, top=0.95, bottom=0.05)
    # plt.tight_layout(w_pad=10)
    # plt.show()
    plt.savefig("/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/Figure_5/5_model_predictions.pdf",
                fmt='pdf')


def plot_null_lfc_distribution(sim_dea, out_path=None):
    null_data = sim_dea.results['ko-wt'].coefficients.stack().reset_index()
    null_data.columns = ['net', 'contrast', 'lfc']
    plt.figure(figsize=(8, 7))
    with sns.axes_style("whitegrid"):
        ax = sns.violinplot(x='contrast', y='lfc', data=null_data, color='0.5')
    ax.set_xticklabels(sim_dea.times)
    ax.set_xlabel('Time')
    ax.set_ylabel('LFC')
    ax.set_title('Null model LFC distribution')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, fmt='pdf')
        plt.close()
    else:
        plt.show()


def make_plots(dde, ts, net_data, sim_dea, matches, tx_to_gene):
    # Get the groups necessary for plots
    ts, censored, n_top, test_sort = calc_groups(ts)
    top = censored.iloc[:n_top]

    null_path = "/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/null_lfc.pdf"
    plot_null_lfc_distribution(sim_dea, out_path=null_path)

    # Plot relationship between KI-WT LFC deviation and prediction
    fig_path = "/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/absdev_vs_diff.pdf"
    print("KI vs error")
    plot_error_predictor(np.log2(censored.abs_dev), censored.grouped_diff, out_path=fig_path)

    print("KI vs KO")
    fig_path = "/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/absdev_vs_meanlfcdev.pdf"
    plot_error_predictor(np.log2(censored.abs_dev), np.log2(censored.mean_abs_lfc), out_path=fig_path)

    print("KO vs error")
    fig_path = "/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/meanlfcdev_vs_diff.pdf"
    plot_error_predictor(np.log2(censored.mean_abs_lfc), censored.grouped_diff,  c='KO', out_path=fig_path)

    # Plot elbow rule
    top_path = "/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/sorting.pdf"
    plot_top_cut(censored, n_top, top_path)

    # Plot the moving median
    rs_path = "/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/running_stats.pdf"
    # plot_running_stat(unsorted, censored, ts, top, out_path=rs_path)

    # Precision recall plot
    pr_path ="/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/aupr.pdf"
    plot_pr(censored, test_sort, n_top, pr_path)

    # Plot the main paneled figure
    panel_plot(ts, top, censored, test_sort, matches, dde, tx_to_gene, sim_dea, net_data)


def main():
    """
     ===================================
     ====== Set script parameters ======
     ===================================
     """
    # todo: argv and parsing

    # Set globals
    sns.set_palette(Bold_8.mpl_colors)
    pd.set_option('display.width', 250)

    # External files
    rna_seq = '../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt'
    compiled_sim = '../data/motif_library/gnw_networks/all_sim_compiled_for_gse69822.pkl'
    gene_names = '../data/GSE69822/mcf10a_gene_names.csv'
    net_path = '../data/motif_library/gnw_networks/simulation_info.csv'

    e_condition = 'ko'  # The experimental condition used
    c_condition = 'wt'  # The control condition used
    t_condition = 'ki'  # The test condition

    override = True  # Rerun certain parts of the analysis

    """
    ===================================
    ========== Load the data ==========
    ===================================
    """

    # NOTE: This process currently assumes some intermediate information is already saved
    # Prep the raw data
    project_name = "GSE69822"

    # Features of the samples taken that are used in calculating statistics
    sample_features = ['condition', 'replicate', 'time']

    # Load sim data
    sim_data = load_sim_data(compiled_sim)

    raw = load_data(rna_seq, sample_features, bg_shift=False)
    tx_to_gene = pd.read_csv(gene_names, index_col=0)

    # Remove unnecessary data
    basic_data = raw.loc[:, [e_condition, t_condition, c_condition]].copy()

    net_data = pd.read_csv(net_path)
    """
        ===================================
        ============= Training ============
        ===================================
    """
    # dde = DDE(project_name)
    # dde.train(project_name, basic_data, sim_data, exp=e_condition,
    #           override=override, correlate=False)
    # # A lot went into the training. We should save the results
    # dde.to_pickle()

    dde = pd.read_pickle('GSE69822/GSE69822_{}-{}_dde.pkl'.format(e_condition, c_condition))    # type: DDE

    matches = dde.match
    sim_dea = dde.sim_dea

    """
        ====================================
        ============= TESTING ==============
        ====================================
    """
    # predictions = dde.predict(t_condition)

    # Calculate testing scores
    # ts = dde.score(t_condition, c_condition)
    # ts.to_pickle('GSE69822/GSE69822_{}-{}_scoring.pkl'.format(e_condition, c_condition))
    ts = pd.read_pickle('GSE69822/GSE69822_{}-{}_scoring.pkl'.format(e_condition, c_condition))

    """
        ====================================
        ============= PLOTTING =============
        ====================================
    """
    make_plots(dde, ts, net_data, sim_dea, matches, tx_to_gene)


if __name__ == '__main__':
    main()