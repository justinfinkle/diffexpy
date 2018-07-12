import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pydiffexp import DEAnalysis, DEPlot, DEResults, cluster_discrete
from pipeline import DynamicDifferentialExpression as DDE
from palettable.cartocolors.qualitative import Bold_8


def load_data(path, bg_shift=True, **kwargs):
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
    return raw_data


if __name__ == '__main__':
    # Set globals
    sns.set_palette(Bold_8.mpl_colors)
    pd.set_option('display.width', 250)

    plot_mean_variance = False

    """
    ===================================
    ========== Load the data ==========
    ===================================
    """
    # Prep the raw data
    project_name = "GSE69822"
    t = [0, 15, 40, 90, 180, 300]
    raw = load_data('../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt')
    ensembl_to_hgnc = pd.read_csv('../data/GSE69822/mcf10a_gene_names.csv', index_col=0)
    hgnc_to_ensembl = ensembl_to_hgnc.reset_index().set_index('hgnc_symbol')

    # Labels that can be used when making DE contrasts used by limma. This helps with setting defaults
    contrast_labels = ['condition', 'time']

    # Features of the samples taken that are used in calculating statistics
    sample_features = ['condition', 'replicate', 'time']
    raw_dea = DEAnalysis(raw, reference_labels=contrast_labels, index_names=sample_features)

    # Remove unnecessary data
    basic_data = raw_dea.raw.loc[:, ['ko', 'ki', 'wt']]

    """
    Clean up the multiindex. This isn't strictly necessary, but makes this a 
    little cleaner to work with. There are some replicate labels that don't match. 
    This is probably based on how batch sequencing was done. We aren't doing 
    paired tests, so it shouldn't matter to change them.
    """

    # idx = basic_data.columns
    # idx = idx.remove_unused_levels()
    # level = 'replicate'
    # level_index = idx.names.index(level)
    # replace_labels = {4: 2, 3: 1}
    # new_labels = [replace_labels[r] if r in replace_labels.keys() else r for r in idx.labels[level_index]]
    # idx = idx.set_labels(new_labels, level=level)
    # idx = idx.remove_unused_levels()
    # replace_levels = {2: 1, 3: 2, 4: 3}
    # new_level_vals = [replace_levels[r] for r in idx.levels[level_index]]
    # idx = idx.set_levels(new_level_vals, level=level)
    # basic_data.columns = idx

    """
        ===================================
        ============= Training ============
        ===================================
    """
    e_condition = 'ko'  # The experimental condition used
    c_condition = 'wt'  # The control condition used
    dde = DDE(project_name)

    override = False  # Rerun certain parts of the analysis

    matches = dde.train(basic_data, project_name, experimental=e_condition,
                        counts=True, override=override)

    g = matches.groupby('true_gene')
    # sys.exit()

    """
        ====================================
        ============= TESTING ==============
        ====================================
    """
    t_condition = 'ki'  # The test condition
    # predictions, error, sim_pred = dde.predict(t_condition, project_name)

    dde.score(project_name, t_condition, c_condition, plot=True)



