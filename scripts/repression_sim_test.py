import os

import pandas as pd
from pydiffexp.gnw import GnwNetwork, tsv_to_dg, mk_ch_dir

jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
base_path = "../data/motif_library/high_repression/"
perturbations = pd.read_csv("../data/motif_library/high_repression/repression.tsv", sep='\t')
sim_settings = base_path + 'settings.txt'

n_nets = 2172

net = 0
save_path = "{}{}/".format(base_path, net)

wt_sim_path = "{}wt_sim/".format(save_path)
ko_sim_path = "{}ko_sim/".format(save_path)
gs_file = "{}{}_goldstandard_signed.tsv".format(save_path, net)
df, dg = tsv_to_dg(gs_file)

# Need to add a "hidden node" that is the perturbation regulator
# dg.add_edge('u', 'x', sign=1)
# Load wt network
g = GnwNetwork(dg, jar_loc, os.path.abspath(save_path), sim_settings, perturbations)     # type: GnwNetwork

g.save_signed_df(filename='{}_goldstandard_signed.tsv'.format(net))
g.draw_graph(filename='{}_network_diagram'.format(net))
o_sbml = g.out_path + '/{}_original.xml'.format(net)
g.transform('{}_original'.format(net), g.signed_path, 4)
g.load_sbml(o_sbml)
g.add_input(sign='-')
g.tree.write("{}{}_modified.xml".format(save_path, net))
g.load_sbml("{}{}_modified.xml".format(save_path, net))
mk_ch_dir(wt_sim_path, ch=False)
g.perturbations.to_csv('{}{}_wt_dream4_timeseries_perturbations.tsv'.format(wt_sim_path, net),
                       sep='\t', index=False)
g.tree.write("{}{}_wt.xml".format(save_path, net))
g.simulate_network(os.path.abspath("{}{}_wt.xml".format(save_path, net)), save_dir=wt_sim_path)

# Make ki sbml
ko_tree = g.tree.make_ko_sbml('G')
ko_tree.write("{}{}_ko.xml".format(save_path, net))

# Save simulation
mk_ch_dir(ko_sim_path, ch=False)
g.perturbations.to_csv('{}{}_ko_dream4_timeseries_perturbations.tsv'.format(ko_sim_path, net),
                       sep='\t', index=False)
g.simulate_network(os.path.abspath("{}{}_ko.xml".format(save_path, net)), save_dir=ko_sim_path)
