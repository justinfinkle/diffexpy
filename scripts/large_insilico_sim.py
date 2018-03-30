import numpy as np
import pandas as pd
from gnw.simulation import GnwNetwork, mk_ch_dir
from pydiffexp.gnw import tsv_to_dg, degree_info

if __name__ == '__main__':
    jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
    base = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/insilico/strongly_connected/Yeast-100.tsv'
    save_path = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/insilico/strongly_connected_2/'
    save_str = "Yeast-100_anon"

    # Load in the base network
    df, dg = tsv_to_dg(base, False)
    nodes = list(dg.nodes())

    # Anonymize the gene names
    mk_ch_dir(save_path, ch=False)
    anon_dict = {n: "G{}".format(i) for i, n in enumerate(nodes)}
    pd.DataFrame.from_dict(anon_dict, orient='index').reset_index().to_csv("{}gene_anonymization.csv".format(save_path),
                                                                           header=False, index=False)
    df.Source = df.Source.map(anon_dict)
    df.Target = df.Target.map(anon_dict)
    anon_path = "{}Yeast-100_anonymized.tsv".format(save_path)
    df.to_csv(anon_path, sep='\t', index=False, header=False)

    # Anonymize the xml
    with open(base.replace('tsv', 'xml'), "r") as fin:
        with open(anon_path.replace('tsv', 'xml'), "w") as fout:
            for line in fin:
                for name, anon in anon_dict.items():
                    line = line.replace(name, anon)
                fout.write(line)

    df, dg = tsv_to_dg(anon_path)
    nodes = list(dg.nodes())
    net_info = degree_info(dg)

    # YMR016C seems like a decent gene to knockout. Not super central, but it has many input/outputs
    ko_gene = anon_dict['YMR016C']

    # Global regulator
    high_out = anon_dict['YKL062W']

    input_node = 'u'
    n_reps = 3
    input_perturbations = np.repeat(np.linspace(0, 1, 5), n_reps)
    perturb = pd.DataFrame(np.zeros((len(input_perturbations), len(nodes))), columns=nodes)
    perturb.insert(0, input_node, input_perturbations)

    net = GnwNetwork(dg, jar_path=jar_loc, out_path='.', settings=("{}/settings.txt".format(save_path)),
                     perturbations=perturb)
    # Save perturbations
    perturb.to_csv("{}perturbations.csv".format(save_path))
    input_types = {'activating': '+', 'deactivating': "-"}

    for stim_type, sign in input_types.items():
        cur_path = "{}{}/".format(save_path, stim_type)
        net.load_sbml(anon_path.replace('.tsv', '.xml'), add_rxn=False)
        # Add an input node
        net.add_input(input_node, high_out, sign=sign)

        # Save the wt data
        mk_ch_dir(cur_path)
        net.set_outpath(cur_path)
        net.save_signed_df(filename='{}_goldstandard_signed.tsv'.format(save_str))
        net.tree.write('{}_wt.xml'.format(save_str))

        # Write the WT, KO, and KI SBMLs
        ko_tree = net.tree.make_ko_sbml(ko_gene)
        ko_tree.write('{}_ko.xml'.format(save_str))
        ki_tree = net.tree.make_ki_sbml(ko_gene)
        ki_tree.write("{}_ki.xml".format(save_str))
        wt_sim_path = cur_path + '/wt_sim_anon/'
        ko_sim_path = cur_path + '/ko_sim_anon/'
        ki_sim_path = cur_path + '/ki_sim_anon/'
        mk_ch_dir(wt_sim_path, ch=False)
        mk_ch_dir(ko_sim_path, ch=False)
        mk_ch_dir(ki_sim_path, ch=False)

        # Write perturbations
        net.perturbations.to_csv('{}{}_wt_dream4_timeseries_perturbations.tsv'.format(wt_sim_path, save_str),
                               sep='\t', index=False)
        net.perturbations.to_csv('{}{}_ko_dream4_timeseries_perturbations.tsv'.format(ko_sim_path, save_str),
                               sep='\t', index=False)
        net.perturbations.to_csv('{}{}_ki_dream4_timeseries_perturbations.tsv'.format(ki_sim_path, save_str),
                               sep='\t', index=False)

        # Simulate
        net.simulate_network(cur_path + '{}_wt.xml'.format(save_str), save_dir=wt_sim_path)
        net.simulate_network(cur_path + '{}_ko.xml'.format(save_str), save_dir=ko_sim_path)
        net.simulate_network(cur_path + '{}_ki.xml'.format(save_str), save_dir=ki_sim_path)