import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = 'Arial'


def volcano_plot(df, p_value=0.05, log2_fc=1):
    nans = df[df.isnull().any(axis=1)]
    df = df.dropna()
    sig = df[(df['adj.P.Val'] <= p_value) & (np.abs(df['logFC']) >= log2_fc)]
    insig = df[~(df['adj.P.Val'] <= p_value) | ~(np.abs(df['logFC']) >= log2_fc)]

    max_y = np.ceil(np.max(np.abs(sig)))
    # if max_fc<10:
    #     rounded_lim = int(2 * np.floor(float(max_fc)/2))
    #     xticks = np.arange(-rounded_lim, rounded_lim+2, 2)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(sig['logFC'], sig['adj.P.Val'], c='b', s=100)
    ax.scatter(insig['logFC'], insig['adj.P.Val'], c='k', s=100)
    ax.set_xlim([-max_fc, max_fc])
    ax.set_ylim([0, np.ceil(np.max(adj_pval))])
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(21, step=2))
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlabel(r'$log_2(\frac{KO}{WT})$', fontsize=28)
    ax.set_ylabel(r'$-log_{10}(pval)$', fontsize=28)
#     boolean_idx = np.arange(len(log2_fc))[(np.abs(log2_fc.values) >= log2_thresh) & (adj_pval.values >= log_odds_thresh)]
#     leftover_idx = np.arange(len(log2_fc))[
#         ~((np.abs(log2_fc.values) >= log2_thresh) & (adj_pval.values >= log_odds_thresh))]
#     x_filtered = x[x > x_cutoff]
#     y_filtered = y[y > y_cutoff]
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.scatter(log2_fc[top_idx], adj_pval[top_idx], c='b', s=100)
#     ax.scatter(log2_fc[remaining_idx], adj_pval[remaining_idx], c='orange', s=100)
#     ax.scatter(log2_fc[leftover_idx], adj_pval[leftover_idx], c='k', s=100)
#     ax.set_xlim([-max_fc, max_fc])
#     ax.set_ylim([0, np.ceil(np.max(adj_pval))])
#     ax.set_xticks(xticks)
#     ax.set_yticks(np.arange(21, step=2))
#     ax.tick_params(axis='both', which='major', labelsize=24)
#     ax.set_xlabel(r'$log_2(\frac{KO}{WT})$', fontsize=28)
#     ax.set_ylabel(r'$-log_{10}(pval)$', fontsize=28)
#
# filepath = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/data/intersection_genes_pvals.csv'
# save_string = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/data/differential_expression/raw_python_volcano_plot2.pdf'
# save_genes = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Python/sprouty/data/ko0_wt0_de_expressed_genes.txt'
#
# results_df = pd.read_csv(filepath)
# new_cols = results_df.columns.values
# new_cols[0] = 'Gene'
# results_df.columns = new_cols
#
# log2_fc = results_df.logFC
# adj_pval = -np.log10(results_df['adj.P.Val'])
# log_odds = results_df.B
#
# # Rename and reformat labels
# results_df.Gene = [(label.replace('X', '').capitalize()).replace('rik', 'Rik') if 'RIK' in label
#                    else (label.capitalize()).replace('rik', 'Rik')
#                    for label in results_df.Gene]
#
# log2_thresh = 1
# log_odds_thresh = 4.6   # Corresponds to 99% of differential expression
#
# # Calculate indices that meet the above criteria
# boolean_idx = np.arange(len(log2_fc))[(np.abs(log2_fc.values)>=log2_thresh) & (adj_pval.values>=log_odds_thresh)]
# leftover_idx = np.arange(len(log2_fc))[~((np.abs(log2_fc.values)>=log2_thresh) & (adj_pval.values>=log_odds_thresh))]
# n_criteria = len(boolean_idx)
#
#
# n_top = 10
# top_idx = boolean_idx[:n_top]
# remaining_idx = boolean_idx[n_top:]
# max_fc = np.ceil(np.max(np.abs(log2_fc)))
# if max_fc<10:
#     rounded_lim = int(2 * np.floor(float(max_fc)/2))
#     xticks = np.arange(-rounded_lim, rounded_lim+2, 2)
#
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(log2_fc[top_idx], adj_pval[top_idx], c='b', s=100)
# ax.scatter(log2_fc[remaining_idx], adj_pval[remaining_idx], c='orange', s=100)
# ax.scatter(log2_fc[leftover_idx], adj_pval[leftover_idx], c='k', s=100)
# ax.set_xlim([-max_fc, max_fc])
# ax.set_ylim([0, np.ceil(np.max(adj_pval))])
# ax.set_xticks(xticks)
# ax.set_yticks(np.arange(21, step=2))
# ax.tick_params(axis='both', which='major', labelsize=24)
# ax.set_xlabel(r'$log_2(\frac{KO}{WT})$', fontsize=28)
# ax.set_ylabel(r'$-log_{10}(pval)$', fontsize=28)
# #for ii, xy in enumerate(zip(log2_fc[top_idx], adj_pval[top_idx])):
# #    plt.annotate(results_df.iloc[ii,0].capitalize(), xy=xy, fontsize=16, style='italic')
# # plt.tight_layout()
# plt.show()
# #plt.savefig(save_string, format='pdf')