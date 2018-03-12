import os

import pandas as pd
from pydiffexp.gnw import GnwNetwork, tsv_to_dg, mk_ch_dir

jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
base_path = "../../data/motif_library/gnw_networks/"
perturbations = pd.read_csv("../../data/motif_library/gnw_networks/perturbations.tsv", sep='\t')
sim_settings = base_path + 'settings.txt'

n_nets = 2172

for net in range(n_nets):
    cur_dir = "{}{}/".format(base_path, net)
    ki_sim_path = "{}ki_sim/".format(cur_dir)
    df, dg = tsv_to_dg("{}{}_goldstandard_signed.tsv".format(cur_dir, net))
    # Only knockin networks for which G has indegree
    if dg.in_degree("G") > 0:
        print(net)
        if os.path.isdir(ki_sim_path):
            print('skip')
            continue
        # Load wt network
        g = GnwNetwork(dg, jar_loc, cur_dir, sim_settings, perturbations)     # type: GnwNetwork
        g.load_sbml("{}{}_wt.xml".format(cur_dir, net))

        # Make ki sbml
        ki_tree = g.tree.make_ki_sbml('G')
        ki_tree.write("{}{}_ki.xml".format(cur_dir, net))

        # Save simulation
        mk_ch_dir(ki_sim_path, ch=False)
        g.perturbations.to_csv('{}{}_ki_dream4_timeseries_perturbations.tsv'.format(ki_sim_path, net),
                               sep='\t', index=False)
        g.simulate_network(os.path.abspath("{}{}_ki.xml".format(cur_dir, net)), save_dir=ki_sim_path)
