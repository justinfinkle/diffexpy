import os
import sys
import copy
import random
import subprocess
import warnings
import xml.etree.ElementTree as eT
import itertools as it
import pandas as pd
import numpy as np
import networkx as nx
from nxpd import draw
from scipy import stats
from typing import Union, Dict

# The namespace needs to be registered before a tree is parsed otherwise it adds ns0
eT.register_namespace('', "http://www.sbml.org/sbml/level2")
sbml_prefix = '{http://www.sbml.org/sbml/level2}'


def mk_ch_dir(path, ch=True):
    """
    Make a directory, or change to it if it already exists
    :param path:
    :return:
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    if ch:
        os.chdir(path)


def module_logic_combos(n_in_edges=2):
    """
    This only works for 2 in edges right now. Can probably be solved recursively
    :param n_in_edges:
    :return:
    """
    if n_in_edges != 2:
        raise ValueError('This method currently only works for 2 input edges')

    mod_sizes = range(1, n_in_edges+1)
    module_logic = [list(it.combinations(mod_sizes, mod_size)) for mod_size in mod_sizes]
    print(module_logic)


def insert_element(e: eT.Element, parent, position, level=None):
    """
    Insert an XML element into another
    :param e:
    :param parent:
    :param position:
    :param level:
    :return:
    """
    if level is not None:
        e.tail = '\n' + '\t'*level
    parent.insert(position, e)


def get_e_levels(element, d=None, level=0) -> dict:
    """
    Recursively get the levels of each element
    :param element:
    :param d:
    :param level:
    :return:
    """
    # Initialize dictionary
    if d is None:
        d = {}

    for child in list(element):
        # Store existing element level
        d[child] = level

        # Add new levels
        get_e_levels(child, d, level + 1)
    return d


def tsv_to_dg(path):
    """
    Read a GNW gold standard tsv and return a dataframe and a DiGraph
    :param path:
    :return:
    """
    df = pd.read_csv(path, sep='\t', header=None, names=['Source', 'Target', 'Sign'])
    dg = nx.from_pandas_dataframe(df, source='Source', target='Target', create_using=nx.DiGraph())
    return df, dg


def degree_info(dg) -> pd.DataFrame:
    """
    Calculate degree information about a network
    """
    out_deg = pd.DataFrame.from_dict(dg.out_degree(), orient='index')
    in_deg = pd.DataFrame.from_dict(dg.in_degree(), orient='index')
    deg = pd.DataFrame.from_dict(dg.degree(), orient='index')
    info = pd.concat([out_deg, in_deg, deg], axis=1)                    # type: pd.DataFrame
    info.columns = ['out', 'in', 'total']
    info['(out-in)/total'] = (info.out-info['in'])/info.total

    return info


def make_perturbations(target, node_list, reps=3):
    """
    Create perturbations for a target node
    """
    node_idx = node_list.index(target)
    positive = np.zeros((reps, len(node_list)))
    positive[:, node_idx] = 1
    negative = positive * -1
    perturbs = np.vstack((positive, negative))
    return perturbs


class RegulatoryModule(object):
    """
    An object to make generating SBMLs easier
    """
    def __init__(self, target, edges: list, sign_str='sign'):
        """
        Expects a list of networkx edges with attributes
        :param edges:
        :param sign_str:
        """
        self.target = target
        self.edges = edges
        self.sign_str = sign_str
        self.activators = []
        self.deactivators = []

        # Get lists
        self.set_activators()
        self.set_deactivators()
        self.enhancer = self.is_enhancer()
        if not self.enhancer:
            self.swap_act_deact()

    def set_activators(self):
        self.activators = [source for source, sign in self.edges if sign == '+']

        # self.activators = [source for source, target, data in self.edges if data['sign'] > 0]

    def set_deactivators(self):
        self.deactivators = [source for source, sign in self.edges if sign == '-']
        # self.deactivators = [source for source, target, data in self.edges if data['sign'] < 0]

    # def correct_module(self):
    #     """
    #     It appears that if there are only deactivators, then the module is set to NOT be an enhancer
    #     and all the deactivators are labeled as activators.
    #     :return:
    #     """
    #     if not self.activators and self.deactivators:
    #         self.enhancer = False
    #         self.activators = self.deactivators.copy()
    #         self.deactivators = []
    #     return

    def swap_act_deact(self):
        """"
        If the module is NOT an enhancer then the activators and deactivators are swapped
        """
        activators = self.activators.copy()
        self.activators = self.deactivators.copy()
        self.deactivators = activators

    def is_enhancer(self):
        if len(self.activators) > len(self.deactivators):
            enhancer = True
        elif len(self.activators) < len(self.deactivators):
            enhancer = False
        else:
            enhancer = random.choice([True, False])

        return enhancer

    def to_str(self):
        # There is a weird inconsistency in GNW xml writing. Correct for it here so that it matches
        synth_name = "(" if self.enhancer else "~("

        for ii, source in enumerate(sorted(self.activators+self.deactivators)):
            if source in self.activators:
                if ii != 0:
                    synth_name += "*"
                synth_name += source
            elif source in self.deactivators:
                synth_name += "~{}".format(source)

        # # If there are activators add them first
        # for aa, activator in enumerate(self.activators):
        #     sep = "*"
        #     if aa == len(self.activators)-1:
        #         sep = ""
        #     synth_name += "{}{}".format(activator, sep)
        #
        # # Set default deactivator separator
        # dsep = '~'
        #
        # # If there are no activators, then the module will not be an enhancer and everything inside is safely multiplied
        # if not self.activators:
        #     dsep = '*'
        # for dd, deactivator in enumerate(self.deactivators):
        #     if dd == 0 and dsep == '*':
        #         sep = ""
        #     else:
        #         sep = dsep
        #     synth_name += "{}{}".format(sep, deactivator)

        # Duplicate separators at the edge case shouldn't exist, but remove just in case
        synth_name = synth_name.replace("*~", "~")

        full_name = synth_name + ")"
        return full_name


class SBMLReaction(object):
    """
    Ideally this would be subclassed, but that doesn't seem to be possible.
    """
    def __init__(self, e: eT.Element, df: pd.DataFrame, base_level):
        self.element = copy.deepcopy(e)
        self.base_level = base_level
        self.levels = get_e_levels(e, level=self.base_level)

        # Get reaction pertinent information
        self.node, self.rxn_type = self._get_rxn_info()
        self.df = df.copy()                                     # type: pd.DataFrame
        self.edge_df = self.get_edge_df(df)                     # type: pd.DataFrame
        self.species = self.get_species_order()
        self.n_inputs = len(self.species)
        self.modifiable = True if self.n_inputs > 1 else False

        # Set state variables
        self.name = self._get_name()
        self.alphas = self.get_alphas()
        self.linear = None                                      # type: Union[bool, None]
        self.n_modules = 0                                      # type: int
        self.update_state()

    def _get_rxn_info(self):
        """
        Get species name and reaction type
        :return:
        """
        return tuple(self.element.get('id').split('_'))

    def update_state(self, new_name=None):
        """
        Update the state of the reaction
        :param new_name:
        :return:
        """
        self.levels = get_e_levels(self.element, level=self.base_level)
        self.alphas = self.get_alphas()
        self.linear, self.n_modules = self.is_linear()
        if new_name is not None:
            self._set_name(new_name)
        self.name = self._get_name()

    def get_edge_df(self, df):
        """
        Get just the appropriate subset of edges
        :param df:
        :return:
        """
        return df.loc[df.target == self.node, :]

    def _get_name(self):
        return self.element.get('name')

    def _set_linear(self):
        """
        Set the state of the reaction
        :return:
        """
        self.linear, self.n_modules = self.is_linear()

    def _set_name(self, name):
        """
        Change the name
        :param name: str
        :return:
        """
        self.element.set('name', name)

    def is_linear(self):
        """
        Get the state of the reaction
        :return:
        """
        if len(self.alphas) > 0:
            n_modules = np.log2(len(self.alphas))
        else:
            n_modules = 0
        linear = None
        if n_modules == 1:
            linear = False
        elif n_modules == self.n_inputs:
            linear = True
        return linear, int(n_modules)

    def get_alphas(self):
        """
        Use the  'kineticLaw' from GNW sbml document
        :return:
        """
        # Count the number of parameters in the kinetic law
        return [c for c in self.element.iter() if ('id' in c.attrib) and ('a_' in c.get('id'))]

    def get_species_order(self):
        """
        Get the species in listOfModifiers.
        NOTE: This assumes a set structure in the reaction SBML element
        :param e:
        :return:
        """
        # Get species in proper order
        return tuple([c.get('species') for c in self.element[2]])

    def _switch_type(self):
        """
        Change the type of reaction from linear <--> complex
        NOTE: only works for 2 inputs so equivalent to (1) + (2) <---> (1*2) (subject to module signs)
        :param modules:
        :return:
        """
        # Only trigger on modifiable reactions
        if not self.modifiable:
            warnings.warn('reaction type switch attempted for a non modifiable element')
        else:
            # Set to linear or not
            to_linear = not self.linear
            # Get the edge signs in the correct order. Should always be 1, 2, 3, ...
            edge_signs = self.edge_df.loc[[(s, self.node) for s in self.species], 'value'].values
            if to_linear:
                edges = [[(str(idx+1), sign)] for idx, sign in enumerate(edge_signs)]
            else:
                edges = [[(str(idx+1), sign) for idx, sign in enumerate(edge_signs)]]
            reg_mods = [RegulatoryModule(self.node, e) for e in edges]
            enhancer, rm_str = zip(*[(rm.enhancer, rm.to_str()) for rm in reg_mods])
            new_alphas = self.random_alphas(np.array(enhancer))
            new_name = "{}_synthesis: {}".format(self.node, (" + ".join(rm_str)))
            self._modify_params(reg_mods)
            self._modify_alphas(new_alphas)

            # Change state
            self.update_state(new_name)
        return

    def _modify_params(self, modules):
        """
        Modify additional parameters other than alpha values
        :param num_modules:
        :return:
        """
        num_modules = len(modules)
        binds_as_complex = self.binds_as_complex(num_modules)
        param_list = self._get_reaction_params()

        # If linear --> complex, remove extra parameters
        if self.n_modules > num_modules:
            # Initialize list of elements to remove
            remove_list = []
            for param in param_list:
                name = param.get('name')
                if ('bindsAsComplex' in name) or ('numActivators' in name) or ('numDeactivators' in name):
                    mod_num = int(param.get('name').split('_')[1])
                    # If it is a larger number than the new number of modules remove it
                    if mod_num > num_modules:
                        remove_list.append(param)
                    # Otherwise set the parameters appropriately
                    else:
                        if 'bindsAsComplex' in name:
                            param.set('value', str(binds_as_complex[mod_num-1]))
                        elif 'numActivators' in name:
                            param.set('value', str(len(modules[mod_num-1].activators)))
                        elif 'numActivators' in name:
                            param.set('value', str(len(modules[mod_num-1].deactivators)))
            # Now remove
            for e in remove_list:
                param_list.remove(e)

        # Else if complex --> linear, add extra parameters
        else:
            if num_modules == self.n_modules:
                warnings.warn('parameter modification called without any necessary changes')

            alpha_start = self._find_param_start('a_0')
            for mod_num in range(1, num_modules+1):
                module_param = ['bindsAsComplex_{}'.format(mod_num),
                                'numActivators_{}'.format(mod_num),
                                'numDeactivators_{}'.format(mod_num)]
                for mod_param in module_param:
                    p = param_list.find("*[@name='{}']".format(mod_param))
                    if p is not None:

                        # Modify values
                        if 'bindsAsComplex' in mod_param:
                            p.set('value', str(binds_as_complex[mod_num - 1]))
                        elif 'numActivators' in mod_param:
                            p.set('value', str(len(modules[mod_num - 1].activators)))
                        else:
                            p.set('value', str(len(modules[mod_num - 1].deactivators)))
                    else:

                        # Make a new element
                        if 'bindsAsComplex' in mod_param:
                            value = binds_as_complex[mod_num - 1]
                        elif 'numActivators' in mod_param:
                            value = len(modules[mod_num - 1].activators)
                        else:
                            value = len(modules[mod_num - 1].deactivators)
                        new_element = eT.Element('parameter', {'name': mod_param,
                                                               'id': mod_param,
                                                               'value': str(value)})

                        # Subtract 1 from the parent tree level for correct offset

                        insert_element(new_element, param_list, alpha_start-1, level=self.levels[param_list])

                    # Update the insert position
                    alpha_start = self._find_param_start('a_0')
        return

    def _modify_alphas(self, new_alphas):
        """
        Change the alpha values as necessary. Assumes new_alphas and self.alphas are correctly ordered lists
        :param new_alphas:
        :return:
        """
        n_alphas = len(self.alphas)
        n_new = len(new_alphas)
        param_list = self._get_reaction_params()
        a_start = self._find_param_start('a_0')
        if n_new > n_alphas:
            # Write values and add new ones
            for aa, new_alpha in enumerate(new_alphas):
                name = "a_{}".format(aa)
                if aa < n_alphas:
                    # Confirm name
                    if self.alphas[aa].get('name') == name:
                        self.alphas[aa].set('value', str(new_alpha))
                else:
                    # Insert a new element at the aa position offset from the alpha start values in the list
                    new_element = eT.Element('parameter', {'name': name, 'id': name, 'value': str(new_alpha)})

                    # Subtract 1 from the parent tree level for correct offset
                    insert_element(new_element, param_list, a_start+aa, level=self.levels[param_list])

        elif n_new < n_alphas:
            # Write values and remove old ones
            for aa, alpha in enumerate(self.alphas):
                if aa < n_new:
                    alpha.set('value', str(new_alphas[aa]))
                else:
                    param_list.remove(alpha)

        else:
            # This shouldn't happen if chaning modes, but just write the values
            for aa, alpha in enumerate(new_alphas):
                self.alphas[aa].set('value', str(alpha))
        return

    def _get_reaction_params(self):
        """
        Get the listOfParameters for the reaction
        :return:
        """
        return [e for e in self.element.iter() if 'kinetic' in e.tag][0][0]

    def _find_param_start(self, value, attrib='name'):
        """
        Get the starting index for the alpha values
        """
        params = self._get_reaction_params()
        a_start = 0
        for idx, param in enumerate(params.iter()):
            if param.get(attrib) == value:
                a_start = idx
                break
        return a_start

    def switch_rxn_type(self):
        """
        Return a new reaction with the opposite kinetics
        :return:
        """
        new_rxn = self.copy()
        new_rxn._switch_type()
        return new_rxn

    def copy(self):
        return SBMLReaction(self.element, self.df, self.base_level)

    """
      Methods that simulate GNW kinetic parameter creation
      """

    @staticmethod
    def random_parameter_gaussian(lower, upper, mean=None, stdev=None):
        """
        Makes a truncated gaussian to mimic GNW
        :param lower:
        :param upper:
        :param mean:
        :param stdev:
        :return:
        """
        if mean is None:
            mean = (lower + upper) / 2
        if stdev is None:
            stdev = (upper - lower) / 6

        return stats.truncnorm((lower - mean) / stdev, (upper - mean) / stdev, loc=mean, scale=stdev)

    @staticmethod
    def binds_as_complex(num_modules) -> list:
        """
        Return a random boolean vector to describe if modules bind as a complex
        :param num_modules:
        :return:
        """

        return list(np.random.randint(0, 2, num_modules).astype(str))

    def random_alphas(self, is_enhancer):
        """
        Meant to replicate randomInitializationOfAlphas() method in GNW HillGene class
        :param  is_enhancer: list-like; ordered signs of reaction module enhancer status consisting of "+" and "-"
        :return:
        """

        # Base activation functions
        weak_activation = 0.25  # Hardcoded in GnwSettings.getInstance().getWeakActivation();
        dalpha_dist = self.random_parameter_gaussian(weak_activation, 1)
        low_basal = self.random_parameter_gaussian(0, weak_activation, 0, 0.05)
        medium_basal = self.random_parameter_gaussian(weak_activation, 1 - weak_activation)

        num_modules = len(is_enhancer)
        num_states = 2 ** num_modules
        is_enhancer = np.ones(num_modules) - (2 * ~is_enhancer)  # type: np.array

        # Initialize dalpha vector; the deltas for the alphas in each module
        dalpha = dalpha_dist.rvs(num_modules) * is_enhancer
        alpha = np.zeros(num_states)  # Initialize alpha vector
        max_delta_positive = sum(dalpha[dalpha > 0])
        max_delta_negative = sum(dalpha[dalpha < 0])

        # The first alpha is the basal activation when no upstreams are around
        if num_modules == 0:
            # No modules
            alpha[0] = 1
        elif max_delta_positive == 0:
            # No enhancer modules
            alpha[0] = 1
        elif max_delta_negative == 0:
            # No non enhancer modules
            alpha[0] = low_basal.rvs(1)[0]
        else:
            # Mixed enhancer type modules
            alpha[0] = medium_basal.rvs(1)[0]

        if (max_delta_positive > 0) and ((alpha[0] + max_delta_positive) < 1):
            # Likely triggered when there are no non enhancers
            # print('weird positive condition triggered', max_delta_positive, alpha[0], alpha[0]+max_delta_positive)
            min_pos = 0
            idx_min_pos = -1
            for i in range(0, num_modules):
                if (dalpha[i] > 0) and (dalpha[0] < min_pos):
                    min_pos = dalpha[i]
                    idx_min_pos = i
            dalpha[idx_min_pos] += 1 - alpha[0] - max_delta_positive

        if (max_delta_negative < 0) and ((alpha[0] + max_delta_negative) > weak_activation):
            # Likely triggered when there are no enhancers
            # print('weird negative condition triggered', max_delta_negative, alpha[0]+max_delta_negative)
            min_pos = -1
            idx_min_pos = -1
            for i in range(0, num_modules):
                if (dalpha[i] < 0) and (dalpha[0] > min_pos):
                    min_pos = dalpha[i]
                    idx_min_pos = i
            dalpha[idx_min_pos] += -alpha[0] - max_delta_negative + low_basal.rvs(1)[0]

        # Generate alpha values
        '''
        Alpha values are calculated for each module. States represent combinations of modules.
        '''
        for i in range(1, num_states):
            alpha[i] = alpha[0]
            s = '{0:b}'.format(i)
            for j in range(0, num_modules):
                if ((len(s) - j - 1) >= 0) and (s[len(s) - j - 1] == '1'):
                    alpha[i] += dalpha[j]

            if alpha[i] < 0:
                alpha[i] = 0
            elif alpha[i] > 1:
                alpha[i] = 1

        return alpha


class SBMLTree(eT.ElementTree):
    """
    Interface with goofy GNW SBML Files
    """
    def __init__(self, tree_root, df):
        super().__init__(tree_root)
        self.root = self.getroot()                                      # type: eT.Element
        self.levels = get_e_levels(self.root)
        self.net_df = df                                                # type: pd.DataFrame
        self.nodes = self.get_nodes()
        self.rxn = self.get_rxn()

    def get_nodes(self, xpath="*//*[@compartment='cell']") -> tuple:
        """
        Get the list of species.

        :return:
        """
        return tuple([s.get('id') for s in self.root.findall(xpath) if 'void' not in s.get('id')])

    def get_rxn(self, id_str=None) -> Dict[str, SBMLReaction]:
        """
        Find all the reactions in the SBML tree
        :param id_str:
        :return:
        """
        rxn = self.root.findall("*//*[@reversible='false']")
        if id_str:
            rxn = [r for r in rxn if id_str in r.get('id')]
        rxn = {r.get('id'): SBMLReaction(r, self.net_df, self.levels[r]) for r in rxn}
        return rxn

    def make_ko_sbml(self, node, max_val=0.0000001):
        """
        Make a knockout version of the dynamical model. Set the max value (transcription rate) for the node (to 0)

        Note: this is extremely dependent on the structure of the SBML (XML).
        :param node: str; node to knockout
        :param max_val: float(optional); value to set maximum transcription rate to. Default (0) prohibits any transcription
        :return:
        """
        new_tree = self.copy()
        try:
            target_reaction = node + '_synthesis'
            for v in new_tree.iterfind("*//*[@id='{}']".format(target_reaction)):
                for param in v.iterfind("*//*[@name='max']"):
                    param.set('value', str(max_val))
        except AttributeError as e:
            raise ValueError('No SBML loaded') from e
        return new_tree

    def copy(self):
        return SBMLTree(copy.deepcopy(self.root), self.net_df.copy())


class GnwNetwork(nx.DiGraph):
    """
    A network class that can interface with networkx, gnw-command-line, and external files

    # Note: The GNW command line isn't a great interface and doesn't always do paths well, so the working directory may
    be changed
    """
    def __init__(self, dg: nx.DiGraph, jar_path: str, out_path: str, settings: str, perturbations: pd.DataFrame):
        super().__init__(data=dg)
        self.out_path = out_path
        self.rxn_alphas = random
        self.signed_path = None
        self.original_sbml = None               # type: str
        self.linear_sbml = None                 # type: str
        self.ko_sbml = {}                       # type: dict
        self.tree = None                        # type: SBMLTree

        self.net_df = self.digraph_to_signed_df()

        # Setup defaults for calling gnw jar
        if os.path.isfile(jar_path):
            self.jar_call = ['java', '-jar', jar_path]
        else:
            raise IOError(('File %s does not exist' % jar_path))
        self.settings_default = os.path.abspath(settings)
        self.devnull = open(os.devnull, 'w')
        self.perturbations = perturbations      # type: pd.DataFrame

        self._possible_rxn = None
        self.rxn_combos = None
        self.combo_order = None
        self.possible_rxn = None    # type: Dict[str, SBMLReaction]

    def set_outpath(self, path):
        """
        Overwrite output path
        :param path:
        :return:
        """
        self.out_path = path

    def draw_graph(self, filename=None, fmt='pdf', show=False, **kwargs):
        if filename is None:
            filename = 'network_diagram'
        filepath = self.out_path + '/' + filename + '.' + fmt
        draw(self, filename=filepath, format=fmt, show=show, **kwargs)

    def digraph_to_signed_df(self, sign_str='sign') -> pd.DataFrame:
        """
        Convert a networkx DiGraph to a signed tsv format for gnw

        Per gnw manual (http://gnw.sourceforge.net/manual/gnw-user-manual.pdf):

            "TSV network structure (*.tsv)
                Should be used to save signed network structures. The attribute
                is either ‘+’ (enhancing), ‘−’ (inhibitory), ‘+−’ (dual), ‘?’ (unknown),
                or ‘0’ (zero). Note, if the value for the attribute is omitted,
                the interaction is assumed to be of type unknown
            "
        :param dg: DiGraph;
        :return:
        """
        df = nx.to_pandas_dataframe(self, weight=sign_str)

        # Reshape the dataframe
        df.columns.name = 'target'
        df.index.name = 'source'
        df = df.unstack().reset_index(name='value').sort_index(axis=1)

        # Convert to signed network format
        df = df[df.value != 0]
        df.loc[df.value == 1, 'value'] = '+'
        df.loc[df.value == -1, 'value'] = '-'
        df.index = list(zip(df.source, df.target))

        return df

    def save_signed_df(self, out_dir=None, filename=None, sep='\t', header=False, index=False):
        """
        Save a signed dataframe to the expected TSV format
        :param df: DataFrame; signed dataframe to save
        :param out_dir: str; path to save the file
        :param filename: str; name to save the file
        :param sep: str (optional); separator to use. Default ('\t') for tabs, unlikely to be modified as GNW expects TSV
        :param header: bool (optional); include column labels. Default (False) not included in TSV for GNW
        :param index:  bool (optional); include row labels. Default (False) not included in TSV for GNW
        :return:
        """
        if out_dir is None:
            out_dir = self.out_path

        # Normalize the path
        out_dir = os.path.normpath(out_dir)
        if filename is None:
            filename = 'goldstandard_signed.tsv'

        savefile = os.path.normpath(out_dir + '/' + filename)
        self.net_df.to_csv(savefile, sep=sep, header=header, index=index)
        self.signed_path = savefile

    def gnw_call(self, call_list, stdout=None, stderr=subprocess.PIPE, **kwargs):
        if stdout is None:
            stdout = self.devnull
        if stderr is None:
            stderr = self.devnull
        p = subprocess.Popen(self.jar_call + call_list, stdout=stdout, stderr=stderr, **kwargs)
        output, err = p.communicate()

        if err is not None:
            err = err.decode('utf-8')

        if p.returncode or ((err is not None) and ('Exception' in err)):
            raise Exception(err)

    def simulate_network(self, network_file, save_dir=None, network_name=None, settings=None):
        if settings is None:
            settings = self.settings_default
        call_list = ['--simulate', '-c', settings, '--input-net', network_file]
        if network_name is not None:
            call_list += ['--network-name', network_name]
        end_dir = os.getcwd()
        if save_dir is not None:
            # Temporarily change the directory because the GNW flag for the output directory doesn't seem to work
            os.chdir(save_dir)

        # Make call to GNW
        self.gnw_call(call_list)

        # Reset the directory
        os.chdir(end_dir)

    def transform(self, network_name, in_path, out_fmt, out_path=None, settings=None, keep_self=True):
        """
        Transform one network file into another type
        """
        end_dir = os.getcwd()
        if not out_path:
            out_path = self.out_path

        # Temporarily change the directory because the GNW flag for the output directory doesn't seem to work
        os.chdir(out_path)

        if settings is None:
            settings = self.settings_default

        call_list = ['--transform', '-c', settings, '--input-net', in_path,
                     ('--output-net-format=' + str(out_fmt)), '--network-name', network_name]
        if keep_self:
            call_list.append('--keep-self-interactions')

        # Make call to GNW
        self.gnw_call(call_list, stderr=None)

        if out_fmt == 4:
            # Set the location of the sbml path
            self.original_sbml = os.path.join(self.out_path, network_name + '.xml')
            self.load_sbml()

        os.chdir(end_dir)

    def load_sbml(self, sbml_path=None, add_rxn=True):
        if sbml_path is None:
            sbml_path = self.original_sbml
        else:
            self.original_sbml = sbml_path
        self.tree = SBMLTree(eT.parse(sbml_path).getroot(), self.net_df)
        if add_rxn:
            self.add_rxn()

        # Reorder perturbations appropriately
        self.perturbations = self.perturbations[list(self.tree.nodes)]  # type: pd.DataFrame

    def add_rxn(self):
        self._possible_rxn = self.make_possible_rxn()
        self.rxn_combos = self._get_rxn_combos()
        self.combo_order = list(self._possible_rxn.keys())
        self.possible_rxn = self._flatten_possible_rxn()

    def make_possible_rxn(self):
        """
        Get the reaction combos that need to be created
        :return:
        """

        possible_reactions = {}
        lookup = {True: 'linear', False: 'multiplicative'}
        for r in self.tree.rxn.values():
            if r.rxn_type == 'synthesis':
                possible_reactions[r.node] = {}
                if r.modifiable:
                    l = lookup[r.linear]
                    possible_reactions[r.node]['{}_{}'.format(r.node, l)] = r
                    # Swap type
                    new_r = r.switch_rxn_type()
                    l = lookup[new_r.linear]
                    possible_reactions[new_r.node]['{}_{}'.format(new_r.node, l)] = new_r
                else:
                    possible_reactions[r.node]['{}'.format(r.node)] = r
        return possible_reactions

    def _get_rxn_combos(self):
        """
        Get possible reaction logic combinations
        :return:
        """
        return list(it.product(*self._possible_rxn.values()))

    def _flatten_possible_rxn(self):
        """
        Flatten the possible reaction dictionary into something easily queried
        :return:
        """
        return {k: v for d in self._possible_rxn.values() for k, v in d.items()}

    def modify_tree_rxns(self, rxn_keys) -> SBMLTree:
        tree = self.tree.copy()

        # Replace reactions
        for r in rxn_keys:
            # Change reactions where necessary
            if self.possible_rxn[r].modifiable:
                self.swap_rxn(tree, self.possible_rxn[r])
        return tree

    @staticmethod
    def swap_rxn(tree: SBMLTree, new_rxn: SBMLReaction):
        """
        Modify in place
        :param tree:
        :param new_rxn:
        :return:
        """

        # Reactions parent
        rxn_list = [c for c in tree.root.iter() if 'listOfReactions' in c.tag][0]      # type: eT.Element

        # Find old reaction and delete
        old_rxn = tree.root.find("*//*[@id='{}']".format(new_rxn.element.get('id')))
        rxn_list.remove(old_rxn)

        # Insert new reaction
        # Don't need to include a tail for levels on this insertion
        # The position must match the order
        position = tree.nodes.index(new_rxn.node)
        insert_element(new_rxn.element, rxn_list, 2*position)






