import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pydiffexp.gnw import GnwSimResults, draw_results, get_graph


def display_sim(network, perturbation, times, directory, exp_condition='ko', ctrl_condition='wt'):
    data_dir = '{}/{}/deactivating/'.format(directory, network)
    network_structure = "{}/{}/{}_goldstandard_signed.tsv".format(directory, network, network)

    ctrl_gsr = GnwSimResults(data_dir, network, ctrl_condition, sim_suffix='dream4_timeseries.tsv',
                             perturb_suffix="dream4_timeseries_perturbations.tsv")
    exp_gsr = GnwSimResults(data_dir, network, exp_condition, sim_suffix='dream4_timeseries.tsv',
                            perturb_suffix="dream4_timeseries_perturbations.tsv")

    data = pd.concat([ctrl_gsr.data, exp_gsr.data]).T
    dg = get_graph(network_structure)
    titles = ["x", "y", "PI3K"]
    mapping = {'G': "PI3k"}
    dg = nx.relabel_nodes(dg, mapping)
    draw_results(data, perturbation, titles, times=times, g=dg)
    plt.tight_layout()


if __name__ == '__main__':
    t = [0, 15, 30, 60, 120, 240, 480]
    display_sim(7, 1, t, "../data/motif_library/gnw_networks/")
    plt.show()