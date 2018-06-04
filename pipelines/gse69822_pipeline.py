
from palettable.cartocolors.qualitative import Bold_8
import pandas as pd
import seaborn as sns
from pydiffexp import DynamicDifferentialExpression
from pydiffexp.gnw import mk_ch_dir


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
    sns.set_palette(Bold_8.mpl_colors)
    # Options
    pd.set_option('display.width', 250)
    override = False  # Rerun certain parts of the analysis
    plot_mean_variance = False

    """
    ===================================
    ========== Load the data ==========
    ===================================
    """
    # Prep the raw data
    project_name = "GSE69822"
    contrast = 'ki-wt'
    t = [0, 15, 40, 90, 180, 300]
    raw = load_data('../data/GSE69822/GSE69822_RNA-Seq_Raw_Counts.txt')
    gene_map = pd.read_csv('../data/GSE69822/mcf10a_gene_names.csv', index_col=0)

    mk_ch_dir(project_name, ch=False)

    """
        ===================================
        ============= Training ============
        ===================================
    """
    e_condition = 'ko'  # The experimental condition used
    c_condition = 'wt'  # The control condition used
    dde = DynamicDifferentialExpression(project_name)
    matches = dde.train(raw, project_name, experimental=e_condition,
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



