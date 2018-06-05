import itertools as it
import sys

import networkx as nx
import numpy as np
import pandas as pd
from pydiffexp.gnw.sim_explorer import tsv_to_dg, to_gephi
import seaborn as sns
from palettable.cartocolors import qualitative
import matplotlib.pyplot as plt


def expected_edges(node_dict, net: nx.DiGraph):
    possible_edges = list(it.permutations(node_dict.values(), 2))
    true_graph = nx.DiGraph()
    for edge in possible_edges:
        if nx.has_path(net, edge[0], edge[1]):
            true_graph.add_edge(edge[0], edge[1])
    return true_graph


if __name__ == '__main__':
    pd.set_option('display.width', 250)

    sim_path = '../data/motif_library/gnw_networks/'
    base = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/insilico/strongly_connected/Yeast-100.tsv'
    df, dg = tsv_to_dg(base)
    edge_dict = {'x': 'YKL062W', 'G':'YMR016C'}
    try:
        print(list(nx.all_simple_paths(dg,'YPR065W', 'YKL062W')))
    except:
        print('no_path')

    motif_matching = pd.read_pickle('intermediate_data/strongly_connected_motif_match.pkl')  # type: pd.DataFrame
    print(motif_matching)

    # filter out non_interesting genes and save to gephi
    filtered_nodes = list(set(motif_matching.true_gene))+list(edge_dict.values())
    keep_parent = np.array([n in filtered_nodes for n in df.Source])
    keep_child = np.array([n in filtered_nodes for n in df.Target])
    df = df[keep_parent & keep_child]
    # to_gephi(df, "intermediate_data/strongly_connected_filtered.csv")
    # sys.exit()
    grouped = motif_matching.groupby('true_gene')
    ii = 0
    max_edges = 6
    summary = pd.DataFrame()
    for gene, data in grouped:
        edge_dict['y'] = gene
        tg = expected_edges(edge_dict, dg)
        te = len(tg.edges())
        ne = max_edges-te
        gene_stats = []
        for row, info in data.iterrows():
            # print(info['id'])
            _, net = tsv_to_dg("{path}/{id}/{id}_goldstandard_signed.tsv".format(path=sim_path, id=info['id']))
            tp = []
            fp = 0
            net_edges = [(edge_dict[e[0]], edge_dict[e[1]]) for e in net.edges()]
            for edge in net_edges:
                try:
                    tp.append(nx.shortest_path_length(dg, edge[0], edge[1]))
                except:
                    fp += 1
            fn = 0
            for edge in tg.edges():
                if edge not in net_edges:
                    fn += 1
            tn = max(0, ne-fn)
            mean_true_path_length = np.mean(tp)
            ntp = len(tp)
            gene_stats.append([ntp/te, ntp+fp, fp/(ntp+fp), 2*ntp/(2*ntp+fp+fn), (ntp+tn)/(ntp+fp+fn+tn),mean_true_path_length])
        gene_data = pd.DataFrame(gene_stats, index=data['id'].values, columns=['TPR', 'total_edges', 'FDR', 'F1', 'ACC', 'mean_tp_length'])
        gene_mean = pd.DataFrame(gene_data.mean())
        gene_mean.columns = [gene]
        gene_mean = gene_mean.T
        gene_mean['true_edges'] = te
        summary = pd.concat([summary, gene_mean], axis=0)
        ii += 1
    print(summary)
    colors = qualitative.Bold_10.mpl_colors
    plt.figure(figsize=(5, 8))
    sns.set_style("whitegrid")
    sns.boxplot(data=summary.loc[:, ['TPR', 'FDR', 'ACC', 'F1']], width=0.5, palette=colors, showfliers=False)
    sns.swarmplot(data=summary.loc[:, ['TPR', 'FDR', 'ACC', 'F1']], color='0.2')
    # plt.plot([-1,3], [0.5, 0.5], '--', c=colors[2], label='Random')
    plt.ylim([0, 1])
    # plt.xticks(rotation='vertical')
    # plt.ylabel('Value')
    plt.tight_layout()
    plt.show()
    sys.exit()

