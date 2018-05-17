import itertools as it
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydiffexp import gnw, DynamicDifferentialExpression


def load_insilico_data(path, conditions, stimuli, net_name, times=None) -> pd.DataFrame:
    """

    :param path:
    :param conditions:
    :param stimuli:
    :param net_name:
    :param times:
    :return:
    """
    # List comprehension: for each combo of stimuli and conditions make a GSR object and get the data
    df_list = []
    for ss, cc in it.product(stimuli, conditions):
        c_df = gnw.GnwSimResults('{}/{}/'.format(path, ss), net_name, cc, sim_suffix='dream4_timeseries.tsv',
                                   perturb_suffix="dream4_timeseries_perturbations.tsv", censor_times=times).data
        if ss == 'deactivating':
            c_df.index = c_df.index.set_levels(-c_df.index.levels[c_df.index.names.index('perturbation')],
                                               'perturbation')
        df_list.append(c_df)

    # Concatenate into one dataframe
    insilico_data = pd.concat(df_list)

    return insilico_data.sort_index()


def plot_data(data, hplot=True, hm=True, plot_log2=True):
    # log2 data for plotting
    plot_data = data.copy()
    if plot_log2:
        plot_data = np.log2(plot_data)

    # Mean-variance plot
    if hplot:
        plt.plot(plot_data.mean(axis=1), plot_data.std(axis=1), '.')
        plt.xlabel('Mean expression')
        plt.ylabel('Expression std')
        plt.title('Heteroskedasticity')
        plt.tight_layout()
        plt.show()

    if hm:
        pass


if __name__ == '__main__':
    # Options
    pd.set_option('display.width', 250)
    override = True     # Rerun certain parts of the analysis
    plot_mean_variance = True
    heatmap = True

    """
    ===================================
    ========== Load the data ==========
    ===================================
    """

    # Set project parameters
    t = [0, 15, 30, 60, 120, 240, 480]
    conditions = ['ko', 'wt', 'ki']
    stimuli = ['activating', 'deactivating']
    reps = 3
    project_name = 'insilico_strongly_connected'
    data_dir = '../data/insilico/strongly_connected/'

    # Keep track of the gene names
    gene_names = pd.read_csv("{}gene_anonymization.csv".format(data_dir), header=None, index_col=0)
    ko_gene = 'YMR016C'
    stim_gene = 'YKL062W'
    ko_gene = gene_names.loc[ko_gene, 1]
    stim_gene = gene_names.loc[stim_gene, 1]
    print(ko_gene, stim_gene)

    # Keep track of the perturbations
    perturbations = pd.read_csv("{}perturbations.csv".format(data_dir), index_col=0)
    p_labels = perturbations.index.values
    n_timeseries = len(p_labels) / reps
    df, dg = gnw.tsv_to_dg("{}Yeast-100_anonymized.tsv".format(data_dir))

    # Get the data an only train when the stimulus perturbation is 1
    raw_data = load_insilico_data(data_dir, conditions, stimuli, 'Yeast-100_anon', times=t)
    idx = pd.IndexSlice
    perturb = 1
    clean_data = raw_data.loc[idx[:, :, perturb, :], :].T      # type: pd.DataFrame

    # Set names appropriately
    clean_data.columns.set_names(['replicate', 'perturbation', 'time'], [1, 2, 3], inplace=True)

    # Shift the background to prevent negatives in log. Only necessary because of GNW data scale
    clean_data += 1

    """
        ===================================
        ============= Training ============
        ===================================
    """
    contrasts = {'train': 'ko-wt', 'test': 'ki-wt'}
    dde = DynamicDifferentialExpression(project_name)
    print(dde.train(clean_data, contrasts['train'], project_name))
    sys.exit()

    prefix = "{}/{}_{}_".format(project_name, project_name, contrasts['train'])  # For saving intermediate data
    g = train(clean_data, contrasts['train'], project_name, prefix, override=override)
    sys.exit()

    # plots = ['515', '1222', '215', '1011']
    # dep = DEPlot()
    # for p in plots:
    #     d = pl.get_net_data(p, 'activating', '../data/motif_library/gnw_networks/', ['ki'])
    #     pd = d.loc['y', idx['ki', :, 1, t]]
    #     dep.tsplot(pd, subgroup='Time', no_fill_legend=True, mean_line_dict={'color': '#7F3C8D'})
    #     plt.ylim([0, 1])
    #     plt.tight_layout()
    #     plt.savefig('{}G{}_ki.pdf'.format(prefix, p), fmt='pdf')
    #
    # dep = DEPlot()
    # # todo: Functionalize
    # if hm:
    #     de_data = (der.top_table().iloc[:, :7])
    #     sort_idx = dde_genes.sort_values(['Cluster', 'score'], ascending=False).index.values
    #     hm_data = de_data.loc[sort_idx]
    #     hm_data = hm_data.divide(hm_data.abs().max(axis=1), axis=0).multiply(der.p_value.loc[hm_data.index] < 0.05)
    #
    #     cmap = sns.diverging_palette(30, 260, s=80, l=55, as_cmap=True)
    #     plt.figure(figsize=(4, 8))
    #     sns.heatmap(hm_data, xticklabels=False, yticklabels=False, cmap=cmap)
    #     plt.tight_layout()
    #     # plt.savefig('{}dde_genes_heatmap.pdf'.format(prefix), fmt='pdf')
    #     dep.tsplot(dea.data.loc['G46', ['ko', 'wt']], no_fill_legend=True)
    #     plt.ylim([0, 1])
    #     plt.tight_layout()
    #     plt.savefig('{}G46_not_dde.pdf'.format(prefix), fmt='pdf')
    #     dep.tsplot(dea.data.loc['G31', ['ko', 'wt']], no_fill_legend=True)
    #     plt.ylim([0, 1])
    #     plt.tight_layout()
    #     plt.savefig('{}G31_dde.pdf'.format(prefix), fmt='pdf')

    # Plot an example
    eg = g.get_group('G59')
    print(eg)
    sys.exit()
    plots = ['515', '1222', '215', '1011']
    # for p in plots:
    #     pl.display_sim(p, 'activating', 1, t, '../data/motif_library/gnw_networks/', exp_condition='ko', node='y')
    #     plt.ylim([0,1])
    #     plt.savefig('{}G{}_dde.pdf'.format(prefix, p), fmt='pdf')
    #     plt.close()
    #
    # dep.tsplot(dea.data.loc['G59', ['ki']], no_fill_legend=True, mean_line_dict={'color': '#7F3C8D'})
    # plt.ylim([0, 1])
    # plt.tight_layout()
    # plt.savefig('{}G59_ki.pdf'.format(prefix), fmt='pdf')