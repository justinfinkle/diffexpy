import pandas as pd
from gnw.simulation import GnwNetwork
from pydiffexp.gnw import tsv_to_dg


def format_params(listOfParams):
    idx, val = zip(*[(p.get('id'), float(p.get('value'))) for p in listOfParams])
    return pd.Series(data=val, index=idx)

def get_rxn_info(key, rxn):
    info = format_params(rxn._get_reaction_params())
    info['reacion_name'] = rxn.name
    info['input_species'] = rxn.get_species_order()
    info.name = key
    return info

nets = [0, 1, 2, 3]
conditions = ['wt', 'ki', 'ko']
for n in nets:
    for c in conditions:
        print(n, c)
        tsv_path = "/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/example_nets/" \
                   "{}/{}_sim/{}_{}_goldstandard_signed.tsv".format(n, c, n, c)
        jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
        sbml_path = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/example_nets/' \
                    '{}/{}_sim/{}_{}.xml'.format(n, c, n, c)

        df, dg = tsv_to_dg(tsv_path)

        net = GnwNetwork(dg, jar_path=jar_loc, out_path='.', settings="", perturbations="")
        net.load_sbml(sbml_path=sbml_path, add_rxn=False)
        reactions = net.tree.get_rxn()
        k = list(reactions.keys())[0]
        get_rxn_info(k, reactions[k])

        all_rxn = [get_rxn_info(k, r) for k, r in reactions.items()]

        params = pd.concat(all_rxn, axis=1).T
        params.to_pickle('/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/Python/pydiffexp/data/example_nets/'
                         '{}/{}_sim/{}_{}_sbml_params.pkl'.format(n, c, n, c))