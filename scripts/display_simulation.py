import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pydiffexp.gnw import GnwSimResults, get_graph, draw_results

if __name__ == "__main__":
    dea = pd.read_pickle('intermediate_data/strongly_connected_dea.pkl')
    # genes = ['YPR065W']
    # dep = DEPlot()
    # for g in genes:
    #     dep.tsplot(dea.data.loc[g], legend=False, subgroup='Time')
    # plt.tight_layout()

    network = 1713
    data_dir = '../data/motif_library/gnw_networks/{}/'.format(network)
    network_structure = "{}{}_goldstandard_signed.tsv".format(data_dir, network)
    p = 0.75
    t = range(0, 500, 10)
    t = [0, 15, 40, 90, 180, 300]

    wt_gsr = GnwSimResults(data_dir, network, 'wt', sim_suffix='dream4_timeseries.tsv',
                           perturb_suffix="dream4_timeseries_perturbations.tsv")
    ko_gsr = GnwSimResults(data_dir, network, 'ko', sim_suffix='dream4_timeseries.tsv',
                           perturb_suffix="dream4_timeseries_perturbations.tsv")

    data = pd.concat([wt_gsr.data, ko_gsr.data]).T
    dg = get_graph(network_structure)
    titles = ["x", "y", "PTEN"]
    mapping = {'G': "PTEN"}
    dg = nx.relabel_nodes(dg, mapping)
    print(dg.nodes())
    draw_results(data, p, titles, times=t, g=dg)
    plt.tight_layout()
    plt.show()
    sys.exit()
