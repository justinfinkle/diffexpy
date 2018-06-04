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
    # clean_data += 1

    """
        ===================================
        ============= Training ============
        ===================================
    """
    e_condition = 'ko'  # The experimental condition used
    c_condition = 'wt'  # The control condition used
    dde = DynamicDifferentialExpression(project_name)
    matches = dde.train(clean_data, project_name, experimental=e_condition,
                        voom=True)
    g = matches.groupby('true_gene')

    """
        ====================================
        ============= TESTING ==============
        ====================================
    """
    t_condition = 'ki'  # The test condition
    # predictions, error, sim_pred = dde.predict(t_condition, project_name)

    dde.score(project_name, t_condition, c_condition)

    # e = dde.compare_random(t_condition, project_name).reset_index()