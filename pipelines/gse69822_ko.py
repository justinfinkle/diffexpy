from collections import Counter
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from matplotlib.transforms import Affine2D
from palettable.cartocolors.diverging import Earth_7
from palettable.cartocolors.qualitative import Bold_8, Prism_10
from pydiffexp import DEPlot
from pydiffexp.pipeline import DynamicDifferentialExpression as DDE
from pydiffexp.plot import elbow_criteria
from pydiffexp.utils import multiindex_helpers as mi
from scipy import integrate, stats
from sklearn.utils import shuffle


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


def plot_gene_prediction(gene, match, data, sim_der, col_names, ensembl_to_hgnc, ax=None):
    dep = DEPlot()
    matching_sim = match.loc[match.true_gene==gene, 'index'].astype(str).values
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
    ax = dep.tsplot(ts_data, scatter=False, ax=ax, legend=False)
    ax.set_title(ensembl_to_hgnc.loc[gene, 'hgnc_symbol'])
    ax.set_ylabel('')


def regular_points_on_circle(startangle=30, points=3, rad=1):
    sa_rad = startangle*np.pi/180
    rand_angles = np.linspace(0+sa_rad, np.pi*2+sa_rad, num=points+1)[:-1]
    x = np.cos(rand_angles)*rad
    y = np.sin(rand_angles)*rad
    return np.vstack([x,y]).T


def plot_net(ax, node_info, models, labels):
    pal = Earth_7.mpl_colormap
    pie_centers = {labels[n]:point for n, point in enumerate(regular_points_on_circle())}
    pie_rad = 0.3
    pie_colors = [Prism_10.mpl_colors[idx] for idx in [3,5,7]]+['0.5']

    for node, center in pie_centers.items():
        ax.annotate(xy=center, s=node, color='w', ha='center', va='center', fontsize=20)
        ax.pie(node_info[node]['fracs'], startangle=90, radius=pie_rad, center=center, colors=pie_colors)
#         ax.legend(['*', '+', '1'])

    center_lines = {}
    for combo in [('x', 'G'), ('G', 'y'), ('y', 'x')]:
        center_lines[combo] = np.array([pie_centers[combo[0]], pie_centers[combo[1]]])

    astyle = ArrowStyle('simple', head_length=7, head_width=7, tail_width=5)
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

        # Center line if needed
    #     fa = FancyArrowPatch(posA=ss, posB=se, arrowstyle=astyle, lw=0, color='k')
    #     ax.add_artist(fa)

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

        astyle = ArrowStyle('simple', head_length=7, head_width=10, tail_width=7*edge_frac)
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

        astyle = ArrowStyle('simple', head_length=7, head_width=10, tail_width=7*edge_frac)
        fa = FancyArrowPatch(posA=in_s, posB=in_e, arrowstyle=astyle, lw=0, color=ec)
        ax.add_artist(fa)

    ax.axis('equal')
    ax.set_title("n = {}".format(len(models)))

def main():
    # Set globals
    sns.set_palette(Bold_8.mpl_colors)
    pd.set_option('display.width', 250)

    """
    ===================================
    ========== Load the data ==========
    ===================================
    """
    idx = pd.IndexSlice
    # Load sim data
    sim_data = pd.read_pickle('../data/motif_library/gnw_networks/all_sim_compiled_for_gse69822.pkl')
    sim_data = sim_data.loc['y', idx[:, :, 1, :]]
    sim_data.columns = sim_data.columns.remove_unused_levels()
    sim_data.columns.set_names(['replicate', 'time'], level=[1, 3], inplace=True)

    # Prep the raw data
    project_name = "GSE69822"
    t = [0, 15, 40, 90, 180, 300]
    # Features of the samples taken that are used in calculating statistics
    sample_features = ['condition', 'replicate', 'time']

    raw = load_data('../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt', sample_features, bg_shift=False)
    ensembl_to_hgnc = pd.read_csv('../data/GSE69822/mcf10a_gene_names.csv', index_col=0)
    hgnc_to_ensembl = ensembl_to_hgnc.reset_index().set_index('hgnc_symbol')

    # Labels that can be used when making DE contrasts used by limma. This helps with setting defaults
    contrast_labels = ['condition', 'time']

    # Remove unnecessary data
    basic_data = raw.loc[:, ['ko', 'ki', 'wt']]

    # sim_dea = DEAnalysis(sim_data, reference_labels=contrast_labels, index_names=sample_features)
    sim_path = "{}/{}_sim.pkl".format(project_name, project_name)
    # sim_dea.to_pickle(sim_path)
    sim_dea = pd.read_pickle(sim_path)

    """
        ===================================
        ============= Training ============
        ===================================
    """
    e_condition = 'ko'  # The experimental condition used
    c_condition = 'wt'  # The control condition used
    dde = DDE(project_name)

    override = False  # Rerun certain parts of the analysis

    matches = dde.train(basic_data, project_name, sim_dea, experimental=e_condition,
                        counts=True, log2=False, override=override)

    g = matches.groupby('true_gene')

    """
        ====================================
        ============= TESTING ==============
        ====================================
    """
    t_condition = 'ki'  # The test condition

    tr = dde.score(project_name, t_condition, c_condition, plot=False)

    """
        ====================================
        ============= PLOTTING =============
        ====================================
    """

    # Add the LFC data in as a predictor
    der = dde.dea.results['{}-{}'.format(e_condition, c_condition)]
    tr['mean_abs_lfc'] = der.coefficients.loc[tr.index].abs().mean(axis=1)
    tr['percent'] = tr.grouped_diff / tr.random_grouped_e * 100
    unsorted = tr.copy()
    tr.sort_values('mean_abs_lfc', ascending=False, inplace=True)

    # Find predictor of
    print(stats.spearmanr(tr.abs_dev, tr.grouped_diff))
    sns.regplot(np.log2(tr.abs_dev), tr.grouped_diff)
    plt.tight_layout()
    plt.savefig(
        "/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/absdev_vs_diff.pdf",
        fmt='pdf')
    plt.close()

    censored = tr[tr.group_dev < (3 * tr.group_dev.std())]
    kinon = tr[tr.ki_cluster != '(0, 0, 0, 0, 0, 0)']
    n_top, top_val = elbow_criteria(range(len(censored)), censored.mean_abs_lfc)
    plt.plot(range(len(censored)), censored.mean_abs_lfc / censored.mean_abs_lfc.max(), label='sorted mean dev')
    plt.plot([n_top, n_top], [0, 1], 'k', label='cutoff')
    plt.plot([np.sum(censored.iloc[:ii].percent > 0) / ii for ii in range(len(censored))], '.', c='grey',
             label='fraction TP')
    plt.legend()
    plt.savefig("/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/sorting.pdf",
                fmt='pdf')
    plt.close()

    top = censored.iloc[:n_top]
    n_shuff = 100
    x = np.array([shuffle(unsorted.percent.values) for ii in range(n_shuff)])
    stats.wilcoxon(censored.percent)

    rm = np.array([running_stat(shuffle(unsorted.percent.values), n_top, s='median') for ii in range(n_shuff)])
    plt.plot(rm.T, c='0.5', zorder=0, alpha=0.1)
    plt.plot([0, 0], lw=2, c='0.5', label='random_sort', zorder=0)
    plt.plot(np.median(rm, axis=0), label='<random_sort>')
    srm = running_stat(tr.percent.values, n_top, s='median')
    plt.plot([0, len(tr)], [tr.percent.median(), tr.percent.median()], label='med(all models)')
    plt.plot([0, len(tr)], [top.percent.median(), top.percent.median()], label='med(top models)')
    plt.plot(srm, label='LFC sorted')
    plt.plot([0, len(tr)], [0, 0], 'k-', lw=1)
    plt.ylim(-100, 100)
    plt.plot(range(len(censored)), censored.mean_abs_lfc * 10, label='mean_abs_lfc')
    leg = plt.legend(loc='center left', bbox_to_anchor=([1, 0.5]))
    plt.xlim([0, len(srm) - 1])
    plt.tight_layout()
    plt.savefig(
        "/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/SI_figures/running_stats.pdf",
        fmt='pdf')
    plt.close()

    confint = dde.dea.results['ki-wt'].get_confint(dde.dea.times)

    # ### Figure organization

    group_keys = ['full', 'top']
    group_colors = ['0.5'] + Bold_8.mpl_colors[:1]
    color_dict = OrderedDict(zip(group_keys, group_colors))

    print("All % median: {}, % Top median: {}".format(tr.percent.median(), top.percent.median()))
    print()
    print("All % wilcoxp: {}, % Top wilcoxp: {}".format(stats.wilcoxon(tr.percent).pvalue / 2,
                                                        stats.wilcoxon(top.percent).pvalue / 2))
    print()
    print("All ∆ median: {}, ∆ Top median: {}".format(tr.grouped_diff.median(), top.grouped_diff.median()))
    print()
    print("All ∆ wilcoxp: {}, ∆ Top wilcoxp: {}".format(stats.wilcoxon(tr.grouped_diff).pvalue / 2,
                                                        stats.wilcoxon(top.grouped_diff).pvalue / 2))

    ss = tr.sort_values('abs_dev', ascending=False)

    recall = np.cumsum(x > 0, axis=1) / np.sum(x > 0, axis=1)[0]
    precision = np.cumsum(x > 0, axis=1) / (np.arange(x.shape[1]) + 1)
    plt.plot(recall.T, precision.T, '0.5', alpha=0.5)
    plt.plot(np.mean(recall, axis=0), np.mean(precision, axis=0), 'k')

    recall = np.cumsum(censored.percent > 0) / np.sum(censored.percent > 0)
    precision = np.cumsum(censored.percent > 0) / (np.arange(len(censored)) + 1)
    print('KO LFC AUPR: ', integrate.cumtrapz(precision, recall)[-1])
    plt.plot(recall, precision)

    recall = np.cumsum(ss.percent > 0) / np.sum(ss.percent > 0)
    precision = np.cumsum(ss.percent > 0) / (np.arange(len(ss)) + 1)
    print('KI LFC AUPR: ', integrate.cumtrapz(precision, recall)[-1])
    plt.plot(recall, precision)
    plt.close()

    # ### Network plots!

    net_data = pd.read_csv('../data/motif_library/gnw_networks/simulation_info.csv')
    net_data.head()

    plt.figure(figsize=(22, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.4)
    # Create a gridspec within the gridspec. 1 row and 2 columns, specifying width ratio
    gs_left = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.5)

    gs_right = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0],
                                                hspace=0.5, width_ratios=[1, 1, 1, 0.1])
    auroc_ax = plt.subplot(gs_left[0, 0])
    box_ax = plt.subplot(gs_left[1, 0])

    high_pred_ax = plt.subplot(gs_right[0, 0])
    med_pred_ax = plt.subplot(gs_right[0, 1])
    low_pred_ax = plt.subplot(gs_right[0, 2])
    high_net_ax = plt.subplot(gs_right[1, 0])
    med_net_ax = plt.subplot(gs_right[1, 1])
    low_net_ax = plt.subplot(gs_right[1, 2])
    hidden_ax = plt.subplot(gs_right[0, 3])
    hidden_ax.axis('off')
    cbar_ax = plt.subplot(gs_right[1, 3])

    box_data = [tr.percent, top.percent]
    df = pd.concat(box_data, keys=color_dict.keys()).reset_index()
    df.columns = ['dataset', 'gene', 'value']
    df['metric'] = 'percent'

    box_ax = sns.boxplot(data=df, x='metric', y='value', hue='dataset', showfliers=False, width=0.5, notch=True,
                         medianprops=dict(solid_capstyle='butt', color='w'), palette=color_dict,
                         boxprops=dict(linewidth=0), ax=box_ax)
    box_ax.plot([-0.5, len(box_data) - 0.5], [0, 0], 'k-', zorder=0)
    box_ax.set_ylabel('% improvement over random')
    box_ax.set_xlabel('')
    box_ax.legend().remove()

    violin_data = [tr.grouped_diff, top.grouped_diff]
    violin_df = pd.concat(violin_data, keys=color_dict.keys()).reset_index()
    violin_df.columns = ['dataset', 'gene', 'value']
    violin_df['metric'] = 'difference'
    violin_ax = box_ax.twinx()
    c = Bold_8.mpl_colors[2]
    violin_ax = sns.violinplot(data=violin_df, x='metric', y='value', hue='dataset', palette=color_dict,
                               ax=violin_ax, order=[1, 'difference'], inner='stick')
    violin_ax.plot([-0.5, len(violin_data) - 0.5], [0, 0], color=c, zorder=0)
    violin_ax.set_ylabel('improvement over random', color=c, rotation=270, va='bottom')
    violin_ax.tick_params('y', colors=c)
    violin_ax.legend(loc='upper center', ncol=2, bbox_to_anchor=[0.5, 1.2])

    # Set the line widths just for the outer lines on the violinplots
    lw = 0
    from matplotlib.collections import PolyCollection

    for art in violin_ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_linewidth(lw)

    box_ax.set_xticklabels(['%', '∆'])

    """
    AUROC
    """

    auroc_ax.plot(np.cumsum(censored.percent < 0) / np.sum(censored.percent < 0),
                  np.cumsum(censored.percent > 0) / np.sum(censored.percent > 0), label='sorted')
    auroc_ax.plot(np.cumsum(ss.percent < 0) / np.sum(ss.percent < 0),
                  np.cumsum(ss.percent > 0) / np.sum(ss.percent > 0), label='abs_dev')
    auroc_ax.plot((np.cumsum(x < 0, axis=1) / np.sum(unsorted.percent < 0)).T,
                  (np.cumsum(x > 0, axis=1) / np.sum(unsorted.percent > 0)).T, c='0.5', alpha=0.1)
    auroc_ax.plot([0, 1], [0, 1], 'k', label='random')

    auroc_ax.plot([0, 0], [0, 0], lw=2, c='0.5', label='shuffled', zorder=0)
    auroc_ax.set_ylim(0, 1)
    auroc_ax.set_xlim(0, 1)
    auroc_ax.set_yticks([0, 0.5, 1])
    auroc_ax.set_xticks([0, 0.5, 1])
    auroc_ax.set_ylabel('TPRish')
    auroc_ax.set_xlabel('FPRish')
    fpr_cut = (np.cumsum(censored.percent < 0) / np.sum(censored.percent < 0)).iloc[n_top]
    auroc_ax.plot([fpr_cut, fpr_cut], [0, 1], label='top_cut')
    leg = auroc_ax.legend(handlelength=1, loc='center left', bbox_to_anchor=(0.95, 0.5), handletextpad=0.2)

    for line in leg.get_lines():
        line.set_lw(4)
        line.set_solid_capstyle('butt')

    gene = top.index[29]
    plot_gene_prediction(gene, matches, dde.dea.data, sim_dea.results['ki-wt'], sim_dea.times, ensembl_to_hgnc,
                         ax=high_pred_ax)
    high_pred_ax.set_ylabel('Expression')

    labels = ['G', 'x', 'y']
    logics = ['_multiplicative', '_linear', '']
    node_info = {}
    models = net_data.loc[matches[matches.true_gene == gene]['index'].values]
    for node in labels:
        cur_dict = {}
        counts = Counter(models['{}_logic'.format(node)])
        cur_dict['fracs'] = [counts['{}{}'.format(node, log)] for log in logics]
        no_in = sum(models['{}_in'.format(node)] == 0)
        cur_dict['fracs'][-1] -= no_in
        cur_dict['fracs'].append(no_in)
        node_info[node] = cur_dict
    plot_net(high_net_ax, node_info, models, labels)

    gene = tr.sort_values('percent', ascending=False).index[0]
    plot_gene_prediction(gene, matches, dde.dea.data, sim_dea.results['ki-wt'], sim_dea.times, ensembl_to_hgnc,
                         ax=med_pred_ax)

    labels = ['G', 'x', 'y']
    logics = ['_multiplicative', '_linear', '']
    node_info = {}
    models = net_data.loc[matches[matches.true_gene == gene]['index'].values]
    for node in labels:
        cur_dict = {}
        counts = Counter(models['{}_logic'.format(node)])
        cur_dict['fracs'] = [counts['{}{}'.format(node, log)] for log in logics]
        no_in = sum(models['{}_in'.format(node)] == 0)
        cur_dict['fracs'][-1] -= no_in
        cur_dict['fracs'].append(no_in)
        node_info[node] = cur_dict
    plot_net(med_net_ax, node_info, models, labels)

    gene = tr.sort_values('percent', ascending=False).index[30]
    plot_gene_prediction(gene, matches, dde.dea.data, sim_dea.results['ki-wt'], sim_dea.times, ensembl_to_hgnc,
                         ax=low_pred_ax)
    low_pred_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    labels = ['G', 'x', 'y']
    logics = ['_multiplicative', '_linear', '']
    node_info = {}
    models = net_data.loc[matches[matches.true_gene == gene]['index'].values]
    for node in labels:
        cur_dict = {}
        counts = Counter(models['{}_logic'.format(node)])
        cur_dict['fracs'] = [counts['{}{}'.format(node, log)] for log in logics]
        no_in = sum(models['{}_in'.format(node)] == 0)
        cur_dict['fracs'][-1] -= no_in
        cur_dict['fracs'].append(no_in)
        node_info[node] = cur_dict
    plot_net(low_net_ax, node_info, models, labels)
    leg = med_net_ax.legend(['×', '+', 'single input', 'no inputs'], ncol=4, loc='upper center',
                            bbox_to_anchor=(0.5, 0.1), handletextpad=0.5, )
    leg.set_title("Node regulation", prop={'size': 24})

    # Add colorbar
    cmap = Earth_7.mpl_colormap
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                    norm=norm)
    cb1.set_ticks([-1, 0, 1])
    cbar_ax.set_ylabel('Average edge sign')

    plt.tight_layout()
    plt.subplots_adjust(left=0.07, right=0.83, top=0.95)
    plt.show()
    # plt.savefig("/Users/jfinkle/Box Sync/*MODYLS_Shared/Publications/2018_pydiffexp/figures/Figure_5/5_model_predictions.pdf", fmt='pdf')


if __name__ == '__main__':
    main()