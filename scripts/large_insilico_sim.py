import pandas as pd
import numpy as np
import pandas as pd
from gnw.simulation import GnwNetwork, mk_ch_dir
from pydiffexp.gnw.sim_explorer import tsv_to_dg, degree_info, make_perturbations

if __name__ == '__main__':
    jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
    base = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/insilico/strongly_connected/Yeast-100.tsv'
    save_path = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/insilico/strongly_connected/'
    save_str = "Yeast-100"

    # YMR016C seems like a decent gene to knockout. Not super central, but it has many input/outputs
    ko_gene = 'YMR016C'

    # Load in the base network
    df, dg = tsv_to_dg(base)
    nodes = list(dg.nodes())
    net_info = degree_info(dg)

    # Steady state perturbation
    steady = np.zeros((3, len(nodes)))

    # Global regulator
    high_out = 'YKL062W'
    high_perturb = make_perturbations(high_out, nodes)

    # Balanced in and out
    balanced = net_info.abs().idxmin()['(out-in)/total']
    balanced_perturb = make_perturbations(balanced, nodes)

    # Affect a random 10% of nodes
    np.random.seed(8)
    reps = 3
    n_nodes = int(0.1*len(nodes))
    rand_nodes = np.random.randint(0, len(nodes), n_nodes)
    mpositive = np.zeros((reps, len(nodes)))
    mpositive[:, rand_nodes] = 1
    mnegative = mpositive * -1

    half_positive = np.zeros((reps, len(nodes)))
    half_positive[:, rand_nodes[:int(n_nodes/2)]] = 1
    half_positive[:, rand_nodes[int(n_nodes / 2):]] = -1
    perturb = np.vstack((steady, high_perturb, balanced_perturb, mpositive, mnegative, half_positive))
    p_index = ['steady'] * 3 + ['high_pos'] * 3 + ['high_neg'] * 3 + ['balanced_pos'] * 3 + ['balanced_neg'] * 3 + \
              ['multi_pos'] * 3 + ['multi_neg'] * 3 + ['multi_mixed'] * 3
    perturb = pd.DataFrame(perturb, columns=nodes, index=p_index)
    net = GnwNetwork(dg, jar_path=jar_loc, out_path='.', settings=("{}/settings.txt".format(save_path)),
                     perturbations=perturb)
    # Save perturbations
    perturb.to_csv("{}labeled_perturbations.csv".format(save_path))

    net.load_sbml(base.replace('.tsv', '.xml'), add_rxn=False)

    # Save the wt data

    mk_ch_dir(save_path)
    net.set_outpath(save_path)
    net.save_signed_df(filename='{}_goldstandard_signed.tsv'.format(save_str))
    net.tree.write('{}_wt.xml'.format(save_str))

    # Write the WT and KO SBMLs
    ko_tree = net.tree.make_ko_sbml(ko_gene)
    ko_tree.write('{}_ko.xml'.format(save_str))
    wt_sim_path = save_path + '/wt_sim/'
    ko_sim_path = save_path + '/ko_sim/'
    mk_ch_dir(wt_sim_path, ch=False)
    mk_ch_dir(ko_sim_path, ch=False)

    # Write perturbations
    net.perturbations.to_csv('{}{}_wt_dream4_timeseries_perturbations.tsv'.format(wt_sim_path, save_str),
                           sep='\t', index=False)
    net.perturbations.to_csv('{}{}_ko_dream4_timeseries_perturbations.tsv'.format(ko_sim_path, save_str),
                           sep='\t', index=False)
    # Simulate
    net.simulate_network(save_path + '{}_wt.xml'.format(save_str), save_dir=wt_sim_path)
    net.simulate_network(save_path + '{}_ko.xml'.format(save_str), save_dir=ko_sim_path)
