import itertools as it
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pydiffexp import gnw, DynamicDifferentialExpression
from palettable.cartocolors.qualitative import Bold_8
from pydiffexp import pipeline as pl
from scipy import stats
from scipy.spatial.distance import hamming
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


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
    sns.set_palette(Bold_8.mpl_colors)

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
    e_condition = 'ko'  # The experimental condition used
    dde = DynamicDifferentialExpression(project_name)
    matches = dde.train(clean_data, project_name, experimental=e_condition,
                            log2=False)
    g = matches.groupby('true_gene')

    """
        ====================================
        ============= TESTING ==============
        ====================================
    """
    t_condition = 'ki'  # The test condition
    test_dde = DynamicDifferentialExpression(project_name)
    test_matches = test_dde.train(clean_data, project_name, experimental=t_condition,
                             log2=False)
    gt = test_matches.groupby('true_gene')

    # Test for model overlap
    # for gene in set(g.groups.keys()).intersection(gt.groups.keys()):
    #     training_models = set(g.get_group(gene).index.get_level_values('id'))
    #     testing_models = set(gt.get_group(gene).index.get_level_values('id'))
    #     print(training_models.intersection(testing_models))
    # sys.exit()

    # # Predict how the gene will respond compared to the WT
    # try:
    #     true_dea = pd.read_pickle("{}dea.pkl".format(test_prefix))  # type: DEAnalysis
    # except:
    #     true_dea = dde(raw, test_contrast, project_name, save_permute_data=True, calc_p=True, voom=True)
    #
    # predicted_scores = pd.read_pickle("{}dde.pkl".format(test_prefix))
    #
    # try:
    #     pred_sim = pd.read_pickle("{}sim_stats.pkl".format(test_prefix))  # type: pd.DataFrame
    # except:
    #     pred_sim = compile_sim('../data/motif_library/gnw_networks/', times=t,
    #                            save_path="{}sim_stats.pkl".format(test_prefix), experimental='ki')
    idx = pd.IndexSlice
    pred_sim = test_dde.sim_stats.loc[idx[:, 1, 'y'], :].sort_index()
    true_data = test_dde.dea.data.loc[:, idx['ki', :, :]].groupby(level='time', axis=1).mean()

    matches['corr'] = matches.apply(
        lambda x: stats.pearsonr(true_data.loc[x.true_gene], pred_sim.loc[x.name, 'ki_mean'])[0], axis=1)

    pred_wlfc = (pred_sim.loc[:, 'lfc'])  # *(1-pred_sim.loc[:, 'lfc_pvalue']))
    true_wlfc = (test_dde.dea.results['ki-wt'].top_table().iloc[:, :len(t)])  # * (1-true_dea.results['ki-wt'].p_value))

    matches['mae'] = matches.apply(lambda x: mse(true_wlfc.loc[x.true_gene], pred_wlfc.loc[x.name]), axis=1)
    # sns.violinplot(data=match, x='true_gene', y='mae')
    # plt.show()
    # sys.exit()
    pred_clusters = pl.discretize_sim(pred_sim, filter_interesting=False)
    reduced_set = True
    if reduced_set:
        # pred_wlfc = pred_wlfc.loc[~match.index.duplicated()]
        true_wlfc = true_wlfc.loc[list(set(matches['true_gene']))]

    gene_mae_dist_dict = {ii: [mse(pwlfc, twlfc) for pwlfc in pred_wlfc.values] for ii, twlfc in true_wlfc.iterrows()}

    # mae_dist, pear_dist = zip(*[(mae(twlfc, pwlfc), stats.pearsonr(twlfc, pwlfc)) for twlfc in true_wlfc.values for pwlfc in pred_wlfc.values])

    # plt.hist([mae(twlfc, pwlfc) for twlfc in true_wlfc.values for pwlfc in pred_wlfc.values], log=True)
    # plt.show()
    # sys.exit()
    resamples = 100
    g = matches.groupby('true_gene')
    # print(g['mean'].mean())
    # plt.plot(g['mean'].mean(), g['mae'].mean(), '.')
    # plt.show()
    # sys.exit()

    sig = 0
    n_matches = len(g.groups)
    diffs = []


    def sample_stats(df, dist_dict, resamples=100):
        random_sample_means = [np.mean(np.random.choice(dist_dict[df.name], len(df))) for _ in range(resamples)]
        rs_mean = np.median(random_sample_means)
        mean_lfc_mae = mse(true_wlfc.loc[df.name], pred_wlfc.loc[g.get_group(df.name).index].mean(axis=0))
        ttest = stats.mannwhitneyu(df.mae, random_sample_means)
        s = pd.Series([len(df), (rs_mean - df.mae.median()) / rs_mean * 100, ttest.pvalue / 2, mean_lfc_mae, rs_mean,
                       df.mae.median()],
                      index=['n', 'mae_diff', 'mae_pvalue', 'mean_lfc_mae', 'random_mae', 'group_mae'])
        return s


    x = pd.concat([dde.dde_genes, g.apply(sample_stats, gene_mae_dist_dict, resamples)], axis=1).dropna()
    print(x)
    print(stats.mannwhitneyu(x.mean_lfc_mae, x.random_mae))
    print(stats.mannwhitneyu(x.group_mae, x.random_mae))
    plt.figure(figsize=(3,5))   
    melted = pd.melt(x, id_vars=x.columns[:-3], value_vars=x.columns[-3:], var_name='stat', value_name='MAE')
    ax = sns.boxplot(data=melted, x='stat', y='MAE', notch=True, showfliers=False, width=0.5)
    sns.swarmplot(data=melted, x='stat', y='MAE', color='black')
    ax.set(xticklabels=['mean', 'random', 'grouped'])
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel("")
    plt.ylim([-0.5, 8])
    plt.tight_layout()
    plt.show()
    sys.exit()
    sig = x[(x.mae_diff > 0) & (x.mae_pvalue < 0.05)].sort_values('mae_pvalue')
    print(sig)