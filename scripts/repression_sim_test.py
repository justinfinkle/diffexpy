import os

import pandas as pd
from pydiffexp.gnw import GnwNetwork, tsv_to_dg, mk_ch_dir

jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
base_path = "../data/motif_library/high_repression/"
perturbations = pd.read_csv("../data/motif_library/high_repression/repression.tsv", sep='\t')
sim_settings = base_path + 'settings.txt'

n_nets = 2172

net = 0

cur_dir = "{}{}/".format(base_path, net)
wt_sim_path = "{}wt_sim/".format(cur_dir)
ko_sim_path = "{}ko_sim/".format(cur_dir)
df, dg = tsv_to_dg("{}{}_goldstandard_signed.tsv".format(cur_dir, net))
# Load wt network
g = GnwNetwork(dg, jar_loc, cur_dir, sim_settings, perturbations)     # type: GnwNetwork
g.load_sbml("{}{}_wt.xml".format(cur_dir, net))

mk_ch_dir(wt_sim_path, ch=False)
g.perturbations.to_csv('{}{}_wt_dream4_timeseries_perturbations.tsv'.format(wt_sim_path, net),
                       sep='\t', index=False)
g.simulate_network(os.path.abspath("{}{}_wt.xml".format(cur_dir, net)), save_dir=wt_sim_path)

# Make ki sbml
ko_tree = g.tree.make_ko_sbml('G')
ko_tree.write("{}{}_ko.xml".format(cur_dir, net))

# Save simulation
mk_ch_dir(ko_sim_path, ch=False)
g.perturbations.to_csv('{}{}_ko_dream4_timeseries_perturbations.tsv'.format(ko_sim_path, net),
                       sep='\t', index=False)
g.simulate_network(os.path.abspath("{}{}_ko.xml".format(cur_dir, net)), save_dir=ko_sim_path)
