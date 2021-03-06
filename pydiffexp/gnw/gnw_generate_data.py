import functools
import multiprocessing as mp
import operator
import os
import subprocess
import time

import networkx as nx
import pandas as pd
from pydiffexp.gnw.simulation import GnwNetwork, mk_ch_dir


def count_combos(dg):
    combos = {0: 1, 1: 2, 2: 8}
    n_in = [combos[len(dg.in_edges(n))] for n in dg.nodes()]

    return functools.reduce(operator.mul, n_in, 1)


def same_edges(g1: nx.DiGraph, g2: nx.DiGraph):
    """
    Check if 2 digraphs have the same edges
    :param g1:
    :param g2:
    :return:
    """
    return set(g1.edges()) == set(g2.edges())


def iso_index(graph_list, dg):
    """
    Find the index of a graph that is isomorphic with the input graph in a list of unique graphs
    :param graph_list:
    :param dg:
    :return:
    """
    idx = None
    for jj, g in enumerate(graph_list):
        if same_edges(g, dg):
            idx = jj
    return idx


def topo_dict(signed, unsigned):
    """
    Sort a list of signed graphs into a dictionary with the unsigned isomorphs
    :param signed: list;
    :param unsigned: list;
    :return:
    """
    # Initialize the dictionary
    iso_dict = {}
    for ii, ug in enumerate(unsigned):
        # Confirm edges are black for drawing
        nx.set_edge_attributes(ug, 'color', 'black')
        # iso_dict[ii] = {'unsigned': ug, 'signed': []}
        # iso_dict[ii] = []

    # Sort the signed graphs
    for sg in signed:
        # iso_dict[iso_index(unsigned, sg)]['signed'].append(sg)
        iso_dict[sg] = iso_index(unsigned, sg)
    return iso_dict


def simulate(net_file, save_path):
    # Simulate
    if os.path.basename(net_file).replace('.xml', '_dream4_timeseries.tsv') in os.listdir(save_path):
        print(net_file, 'exists')
        return
    print(net_file)
    simulate_network(net_file, save_dir=save_path)

def gnw_call(call_list, stdout=None, stderr=subprocess.PIPE, **kwargs):
    devnull = open(os.devnull, 'w')
    jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
    jar_call = ['java', '-jar', jar_loc]
    if stdout is None:
        stdout = devnull
    if stderr is None:
        stderr = devnull
    p = subprocess.Popen(jar_call + call_list, stdout=stdout, stderr=stderr, **kwargs)
    output, err = p.communicate()

    if err is not None:
        err = err.decode('utf-8')

    if p.returncode or ((err is not None) and ('Exception' in err)):
        raise Exception(err)

def simulate_network(network_file, save_dir=None, network_name=None, settings=None):
    settings_default = '/Volumes/Hephaestus/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/motif_library/gnw_networks/settings.txt'
    if settings is None:
        settings = settings_default
    call_list = ['--simulate', '-c', settings, '--input-net', network_file]
    if network_name is not None:
        call_list += ['--network-name', network_name]
    end_dir = os.getcwd()
    if save_dir is not None:
        # Temporarily change the directory because the GNW flag for the output directory doesn't seem to work
        os.chdir(save_dir)

    # Make call to GNW
    gnw_call(call_list)

    # Reset the directory
    os.chdir(end_dir)

if __name__ == '__main__':
    """
    READ FIRST!
    IMPORTANT NOTE - the jar file doesn't seem to use the --output-path flag appropriately. Files will be saved wherever
    this script is run
    """

    directory = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/motif_library/'
    jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
    jar_call = ['java', '-jar', jar_loc]

    unique_graphs_path = directory + 'unique_wc_3node_signed_noself_nets.pkl'
    unsigned_path = directory + 'unique_wc_3node_unsigned_noself_nets.pkl'
    output_base = directory + 'gnw_networks/'
    sim_settings = output_base + 'settings.txt'
    settings_default = os.path.abspath(sim_settings)
    ko_gene = 'G'

    # Get unique network pickle
    # Expected dictionary of the form {num_edges: [list of networkx graphs]}
    unique_graphs = pd.read_pickle(unique_graphs_path)
    unique_graphs = [graph for sublist in unique_graphs.values() for graph in sublist]

    # Read in the unsigned graphs and make the edge colors black
    unsigned_graphs = [graph for sublist in pd.read_pickle(unsigned_path).values() for graph in sublist]

    perturbation_file = output_base + 'perturbations.tsv'
    p_filename = '_dream4_timeseries_perturbations.tsv'
    graph_dict = topo_dict(unique_graphs, unsigned_graphs)

    perturbations = pd.read_csv(perturbation_file, sep='\t')

    sim_counter = 0
    sim_info = pd.DataFrame()
    start = time.time()
    input_types = {'activating': '+', 'deactivating': "-"}

    # Create a pool of workers
    pool = mp.Pool()

    for ug_num, network in enumerate(unique_graphs):
        save_path = "{}{}".format(output_base, sim_counter)

        # Initialize the gnw graph
        g = GnwNetwork(network, jar_loc, save_path, sim_settings, perturbations)    # type: GnwNetwork
        mk_ch_dir(save_path)

        # Save initial network info
        # n to signed
        #todo: this fails when the number of networks is made is greater than the number of unique graphs
        if os.path.isfile('{}_goldstandard_signed.tsv'.format(sim_counter)):
            print(sim_counter)
            sim_counter += 1
            continue

        g.save_signed_df(filename='{}_goldstandard_signed.tsv'.format(sim_counter))
        g.draw_graph(filename='{}_network_diagram'.format(sim_counter))

        # make sbml
        o_sbml = g.out_path + '/{}_original.xml'.format(sim_counter)
        if not os.path.isfile(o_sbml):
            g.transform('{}_original'.format(sim_counter), g.signed_path, 4)

        try:
            g.load_sbml(g.original_sbml)
        except:
            g.load_sbml(o_sbml)

        # Keep the unmodified tree
        base_tree = g.tree
        for cc, combo in enumerate(g.rxn_combos):
            t = time.time()

            # Change to the correct directory and write the initial data
            save_path = "{}{}/".format(output_base, sim_counter)
            mk_ch_dir(save_path)
            g.set_outpath(save_path)
            edge_path = '{}_goldstandard_signed.tsv'.format(sim_counter)
            dot_path = '{}_network_diagram'.format(sim_counter)
            original_xml = os.path.abspath('{}_original.xml'.format(sim_counter))
            g.tree = base_tree.copy()
            g.save_signed_df(filename=edge_path)
            g.draw_graph(filename=dot_path)
            g.tree.write(original_xml)
            combo_tree = g.modify_tree_rxns(combo)
            combo_order = g.combo_order
            combo_tree.write(original_xml.replace('original', 'combo'))

            starmap_iterable = []
            for stim_type, sign in input_types.items():
                current_path = mk_ch_dir("{}{}/".format(save_path, stim_type))
                # Load original XML
                g.load_sbml(original_xml.replace('original', 'combo'), add_rxn=False)

                # Add the new input node, save the modified xml, and load it for use
                g.add_input(sign=sign)
                g.tree.write("{}{}_{}.xml".format(current_path, sim_counter, stim_type))
                g.load_sbml("{}{}_{}.xml".format(current_path, sim_counter, stim_type), add_rxn=False)

                # Write the wt and ko xmls
                # Make the directory and change to it
                wt_filename = '{}_wt.xml'.format(sim_counter)
                ko_filename = wt_filename.replace('wt', 'ko')

                # Write the WT and KO SBMLs
                g.tree.write(wt_filename)
                ko_tree = g.tree.make_ko_sbml(ko_gene)
                ko_tree.write(ko_filename)
                wt_sim_path = current_path + '/wt_sim/'
                ko_sim_path = current_path + '/ko_sim/'
                mk_ch_dir(wt_sim_path, ch=False)
                mk_ch_dir(ko_sim_path, ch=False)

                # Write perturbations
                g.perturbations.to_csv('{}{}_wt_dream4_timeseries_perturbations.tsv'.format(wt_sim_path, sim_counter),
                                       sep='\t', index=False)
                g.perturbations.to_csv('{}{}_ko_dream4_timeseries_perturbations.tsv'.format(ko_sim_path, sim_counter),
                                       sep='\t', index=False)

                starmap_iterable.append((current_path + wt_filename, wt_sim_path))
                starmap_iterable.append((current_path + ko_filename, ko_sim_path))
                # Simulate
                # g.simulate_network(current_path + wt_filename, save_dir=wt_sim_path)
                # g.simulate_network(current_path + ko_filename, save_dir=ko_sim_path)
            pool.starmap(simulate, starmap_iterable)

            # Save the information
            feature_info = {'net': sim_counter, 'unsigned_group': ug_num,
                            'x_in': 0, 'y_in': 0, '{}_in'.format(ko_gene): 0}
            for source, target, data in g.in_edges(data=True):
                feature_info['{}->{}'.format(source, target)] = data['sign']
                feature_info['{}_in'.format(target)] += 1
            for ii, target in enumerate(combo_order):
                feature_info[target + '_logic'] = combo[ii]
            series = pd.Series(feature_info)

            sim_info = sim_info.append(series, ignore_index=True)

            print(sim_counter, '| completed in {} secs'.format(round(time.time()-t, 3)),
                  '| total time {} min'.format(round((time.time()-start)/60, 3)))
            sim_counter += 1

    # Clean info after all simulations done
    sim_info.set_index('net', inplace=True)
    sim_info.fillna(0, inplace=True)
    sim_info.to_pickle(output_base+'simulation_info.pkl')
    pool.close()
    pool.join()

