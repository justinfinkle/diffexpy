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

tsv_path = "/Users/jfinkle/Desktop/InSilicoSize10-Ecoli1.tsv"
jar_loc = '/Users/jfinkle/Documents/Northwestern/MoDyLS/Code/gnw/gnw-3.1.2b.jar'
sbml_path = '/Users/jfinkle/Desktop/InSilicoSize10-Ecoli1.xml'

df, dg = tsv_to_dg(tsv_path)

net = GnwNetwork(dg, jar_path=jar_loc, out_path='.', settings="", perturbations="")
net.load_sbml(sbml_path=sbml_path, add_rxn=False)
reactions = net.tree.get_rxn()
k = list(reactions.keys())[0]
get_rxn_info(k, reactions[k])

all_rxn = [get_rxn_info(k, r) for k, r in reactions.items()]

params = pd.concat(all_rxn, axis=1).T
params.to_csv('/Users/jfinkle/Desktop/InSilicoSize10-Ecoli1_params.csv')