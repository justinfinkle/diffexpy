import os
import sys
import pickle
import multiprocessing as mp

import pandas as pd
from pydiffexp.gnw import GnwNetwork, tsv_to_dg, mk_ch_dir, simulate

jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
base_path = "../data/motif_library/gnw_networks/"
perturbations = pd.read_csv("../data/motif_library/gnw_networks/perturbations.tsv", sep='\t')
sim_settings = base_path + 'settings.txt'

n_nets = 2172

sim_args = []
count = 0
no_ki_effect_net = []
for net in range(n_nets):
    base_dir = "{}{}/".format(base_path, net)
    df, dg = tsv_to_dg("{}{}_goldstandard_signed.tsv".format(base_dir, net))
    if dg.in_degree("G") == 0:
        count+=1
        no_ki_effect_net.append(net)
print(count)
print(no_ki_effect_net)

with open("{}no_ki_effect_networks.pickle".format(base_path), "wb") as output_file:
    pickle.dump(no_ki_effect_net, output_file)

#     for stim in ['activating', 'deactivating']:
#         cur_dir = "{}{}/".format(base_dir, stim)
#         ki_sim_path = "{}ki_sim/".format(cur_dir)
#         sim_args.append((os.path.abspath("{}{}_ki.xml".format(cur_dir, net)), ki_sim_path))
#         if os.path.isdir(ki_sim_path):
#             print('skip')
#             continue
#         print(net)
#         # Load wt network
#         g = GnwNetwork(dg, jar_loc, cur_dir, sim_settings, perturbations)     # type: GnwNetwork
#         g.load_sbml("{}{}_wt.xml".format(cur_dir, net))
#
#         # Make ki sbml
#         ki_tree = g.tree.make_ki_sbml('G')
#         ki_tree.write("{}{}_ki.xml".format(cur_dir, net))
#
#         # Save simulation
#         mk_ch_dir(ki_sim_path, ch=False)
#         g.perturbations.to_csv('{}{}_ki_dream4_timeseries_perturbations.tsv'.format(ki_sim_path, net),
#                                sep='\t', index=False)
#         # g.simulate_network(os.path.abspath("{}{}_ki.xml".format(cur_dir, net)), save_dir=ki_sim_path)
# #
# with open('intermediate_data/ki_simulation_Args.pkl', 'wb') as f:
#     pickle.dump(sim_args, f)
#
# with open('intermediate_data/ki_simulation_Args.pkl', 'rb') as pickle_file:
#     sim_args = pickle.load(pickle_file)
#
# print(sim_settings)
# pool = mp.Pool()
# pool.starmap(simulate, sim_args)
# pool.close()
# pool.join()