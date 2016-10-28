
import sys
import warnings
import itertools
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import fisher_test as ft
from de_analysis import DEAnalysis
from collections import Counter

########################################################################################################################
########################################################################################################################
# Static Methods
########################################################################################################################
########################################################################################################################


def is_number(s):
    """
    Check if a character is a number
    :param s:
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def clean_index(index):
    """
    Need to clean gene names from R results. R has different defaults for handling strings
    :param index:
    :return:
    """
    cleaned_index = [string[1:].upper().replace('.', '-') if (string[0] == 'X') & (is_number(string[1]))
                     else string.upper().replace('.', '-') for string in index]

    return cleaned_index


def make_binary_combos(char_list, length, kind='char'):
    # Make a iterator
    combo_iterator = itertools.product(char_list, repeat=length)

    # Turn it into a list
    if kind == 'char':
        combos = [list(combo) for combo in combo_iterator]
    else:
        combos = [list(combo) for combo in combo_iterator]
    return combos


def make_binary_trajectories(log2_fc_data, axis=0, threshold=0.1):
    # Take the difference of the data, and remove NaN
    data_diff = log2_fc_data.diff(axis=axis)
    data_diff.fillna(0, inplace=True)

    # Set the threshold
    change_thresh = np.log2(1+threshold)

    # Binarize based on threshold
    data_diff[data_diff >= change_thresh] = 1
    data_diff[data_diff <= -change_thresh] = -1
    data_diff[np.abs(data_diff) < change_thresh] = 0

    # Make a list of the combination tuples
    diff_list = np.array([''.join(row.astype(int).astype(str)) for row in data_diff.values[:, 1:]])
    return diff_list, data_diff


def make_possible_paths(step_list, step_values):
    """
    :param step_list:
    :param step_values:
    :return:
    """
    stepwise_path_combos = {}
    string_to_array_dict = {}
    for step in step_list:
        step_combos = make_binary_combos([0, 1, -1], step, kind='arr')
        stepwise_path_combos[step] = {'0' + ''.join(map(str, combo)): np.array([0] + combo) for combo in step_combos}
        for combo in step_combos:
            string_to_array_dict['0' + ''.join(map(str, combo))]= np.array([0] + combo)
    return string_to_array_dict, stepwise_path_combos


def binary_to_growing_path(step_list, df, joiner=''):
    """

    :param df:
    :return:
    """
    growing_path = pd.DataFrame([], index=df.index)

    for step in step_list:
        a = df.values[:, :step+1].astype(int).astype(str)
        a = [joiner.join(row) for row in a.astype(str)]
        growing_path[step] = a

    return growing_path


def find_genes_in_path(growing_path_df, stepwise_path_combos, gene_dict):

    path_dictionary = {}
    for step, series in growing_path_df.iteritems():
        path_set = set(series)
        path_dictionary[step] = {}
        genes_in_step = 0
        possible_paths = stepwise_path_combos[step]
        for path in path_set:
            path_array = possible_paths[path]
            alt_paths = [alt_path for alt_path in path_set if
                         (alt_path != path) & np.array_equal(possible_paths[alt_path][:step], path_array[:step])]
            genes_in_path = growing_path_df.index[growing_path_df[step]==path].values
            tfs_in_path = ft.convert_gene_to_tf(genes_in_path, gene_dict)
            path_dictionary[step][path] = {'genes': genes_in_path.tolist(), 'tfs': tfs_in_path, 'alt_paths':alt_paths}
            genes_in_step += len(genes_in_path)
        if genes_in_step != len(growing_path_df):
            raise ValueError('Not all genes accounted for in step %i' % step)

    return path_dictionary


def find_path_enrichment(path_dictionary, fdr=0.01):
    print "Searching for enrichment..."
    enriched_path_list = []
    enrichment_table_list = []
    for step, paths in path_dictionary.iteritems():
        for path, value in paths.iteritems():
            study_tfs = value['tfs']
            alts = value['alt_paths']
            # enriched_tfs = []
            tables = []
            background = []
            for alt in alts:
                # background = path_dictionary[step][alt]['tfs']
                background += path_dictionary[step][alt]['tfs']
            table = ft.calculate_study_enrichment(study_tfs, background, fdr=fdr)
            enriched_tfs = table[table['FDR_reject']]['TF'].values.tolist()
                # tables.append(table)
                # enriched_tfs.append(set(table['TF'].values.tolist()))
            # try:
            #     enriched_tfs = list(enriched_tfs[0].intersection(enriched_tfs[1]))
            if enriched_tfs:
                print "Path enriched: ", path, enriched_tfs
                enriched_path_list.append(path)
                enrichment_table_list.append(enriched_tfs)
            # except IndexError:
            #     continue
            # if np.sum(table['FDR_reject']) > 0:
            #     print "Path enriched: ", path
            #     enrichment_table_list.append(table[table['FDR_reject']].sort_values('p_bonferroni'))
            #     enriched_path_list.append(path)
    return enriched_path_list, enrichment_table_list


def plot_enrichment_results(enriched_path_list, string_to_array_dict):
    print len(enriched_path_list)
    n_cols = int(np.floor(np.sqrt(len(enriched_path_list))))
    n_rows = int(np.ceil(np.sqrt(len(enriched_path_list))))
    # f, axarr = plt.subplots(n_rows, n_cols)

    a, b = np.meshgrid(range(n_rows), range(n_cols))
    row_index = a.flatten()
    column_index = b.flatten()
    for ii, sig in enumerate(enriched_path_list):
        found = [string.find(sig) for string in enriched_path_list]
        if np.sum(found)>-len(enriched_path_list)+1:
            print sig
            continue
        plot_data = string_to_array_dict[sig]
        x_ticks = range(len(plot_data))
        max_mag = np.max(np.abs(np.cumsum(plot_data)))
        plt.plot(np.cumsum(plot_data), '.-', lw=3, ms=20, label=sig)
        #plt.plot(x_ticks[-2], plot_data[-2], '.', lw=3, ms=20, c='r')
        # plot_data = string_to_array_dict[sig]
        # max_mag = np.max(np.abs(np.cumsum(plot_data)))
        # x_ticks = range(len(plot_data))
        # axarr[row_index[ii], column_index[ii]].plot(x_ticks, plot_data, '.-', lw=3, ms=20)
        # axarr[row_index[ii], column_index[ii]].plot(x_ticks[-2:], plot_data[-2:], '.-', lw=3, ms=20, c='r')
        # axarr[row_index[ii], column_index[ii]].set_ylim([-max_mag, max_mag])
        # axarr[row_index[ii], column_index[ii]].set_yticks(np.arange(-max_mag, max_mag+1))
        # axarr[row_index[ii], column_index[ii]].set_xticks(x_ticks)
        # axarr[row_index[ii], column_index[ii]].set_xticklabels(times[:len(plot_data)])
    #plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
########################################################################################################################
########################################################################################################################
# Class objects
########################################################################################################################
########################################################################################################################


class DiscretizedClusterer(object):
    """
    Object that handles expression data and clusters it into discretized paths
    """

    def __init__(self, de_analysis_object=None, limma_results_path=None, p_value_threshold=0.05,
                 fold_change_threshold=0.05, fdr=0.01):
        """

        :param de_analysis_object: DEAnalysis or str
        :param limma_results_path: str
        :param p_value_threshold: float
        :param fold_change_threshold: float
        :return:
        """

        self.p_value_threshold = p_value_threshold
        self.fold_change_threshold = fold_change_threshold
        self.limma_results = None
        self.limma_comparisons = None
        self.conditions = None
        self.cluster_dict = None
        self.times = None
        self.enrichment_results = {}
        self.fdr = fdr
        self.string_to_array_dict = {}
        self.up_patches = None
        self.down_patches = None
        self.horizontal_patches = None

        # Try to set a differential expression analysis object. This is needed for plotting data
        # If it doesn't exist, trace data can't be displayed
        if type(de_analysis_object) is DEAnalysis:
            self.de_object = de_analysis_object

        elif type(de_analysis_object) is str:
            try:
                self.de_object = pd.read_pickle(de_analysis_object)
            except ValueError:
                sys.exit(("The selected path %s is not a valid DEAnalysis pickle" % de_analysis_object))

        else:
            warnings.warn('DiscretizedClusterer object initialized without valid DEAnlysis object.'
                          '\nSome methods may not function')

        if limma_results_path is not None:
            self.load_limma_results(limma_results_path)

    def load_limma_results(self, folder_path):
        """
        Load results from R package Limma.
        :param folder_path: str

        :return:
        """

        files = os.listdir(folder_path)

        # Read files into dataframe if it is a csv
        expression_change_results = {}
        for filename in files:
            if '.csv' in filename:
                temp_df = pd.read_csv(folder_path+filename)
                temp_df.index = clean_index(temp_df.index)
                expression_change_results[filename.replace('.csv', '')] = temp_df

        # Save results
        self.limma_results = expression_change_results
        self.limma_comparisons = expression_change_results.keys()
        try:
            self.times = list(set([int(re.split(r"[_-]+", key)[1]) for key in expression_change_results.keys()]).union(
                set([int(re.split(r"[_-]+", key)[-1]) for key in expression_change_results.keys()])))
        except ValueError:
            #todo: this is a catch all for when expresison differences are used
            self.times = [0, 15, 60, 120, 240]
        self.conditions = list(set([re.split(r"[_-]+", key)[0] for key in expression_change_results.keys()]))

    def cluster_genes(self, p_cutoff=None, fold_change_cutoff=None, fold_change_scale='log2'):
        """
        Cluster genes into discretized clusters
        :param p_cutoff:
        :param fold_change_cutoff:
        :param fold_change_scale:
        :return:
        """
        if p_cutoff is None:
            p_cutoff = self.p_value_threshold
        if fold_change_cutoff is None:
            fold_change_cutoff = self.fold_change_threshold
        if fold_change_scale == 'log2':
            fold_change_cutoff = np.log2(1+fold_change_cutoff)

        results_dict = None
        if self.limma_results is not None:
            # Initialize the results
            results_index = self.limma_results.itervalues().next().sort_index().index
            results_dict = {condition: pd.DataFrame([], index=results_index) for condition in self.conditions}

            # Assign each gene to a cluster
            for key, df in self.limma_results.iteritems():

                # Get information to place in the appropriate dataframe
                key_split = re.split(r"[_-]+", key)
                condition = key_split[0]
                try:
                    end_time = int(key_split[1])
                except ValueError:
                    end_time = int(key_split[2])
                results_df = results_dict[condition]

                # Confirm that the list of genes matches
                df.sort_index(inplace=True)
                if not np.array_equal(results_df.index.values, df.index.values):
                    raise Exception("List of sorted genes does not match reference ")

                # Calculate the sign of each fold change comparison
                results_df[end_time] = np.sign(df['logFC'].values)

                # Set values to zero if they don't meet the threshold
                results_df.loc[df['adj.P.Val'] > p_cutoff, end_time] = 0
                results_df.loc[np.abs(df['logFC']) < fold_change_cutoff, end_time] = 0
        # Cleanup the results
        try:
            keep_gene = (((results_dict['WT'].T != 0).any()) | ((results_dict['KO'].T != 0).any())).values
        except KeyError:
            keep_gene = ((results_dict['diff'].T != 0).any()).values

        for key in results_dict.iterkeys():
            results_dict[key][0] = 0
            results_dict[key].sort_index(axis=1, inplace=True, ascending=True)
            results_dict[key] = results_dict[key][keep_gene]

        self.cluster_dict = results_dict.copy()

    def get_enrichment(self, fdr=None):
        association_dict_file = '../clustering/tf_enrichment/gene_association_dict.pickle'
        association_dict = pd.read_pickle(association_dict_file)
        if fdr is None:
            fdr = self.fdr
        else:
            self.fdr = fdr
        for condition in self.conditions:
            print 'Checking enrichment for condtion: %s with FDR of %s' % (condition, str(fdr))
            self.enrichment_results[condition] = {}
            times = self.cluster_dict[condition].columns.values
            steps = range(1, len(times))
            step_values = list(set(self.cluster_dict[condition].values.astype(int).flatten()))
            self.string_to_array_dict, stepwise_path_combos = make_possible_paths(steps, step_values)

            growing_path = binary_to_growing_path(steps, self.cluster_dict[condition])

            path_dictionary = find_genes_in_path(growing_path, stepwise_path_combos, association_dict)

            sig_path_list, enrichment_tables = find_path_enrichment(path_dictionary, fdr=fdr)
            print "%i paths with enrichment found" % len(sig_path_list)
            self.enrichment_results[condition]['significant_paths'] = sig_path_list
            self.enrichment_results[condition]['enrichment_tables'] = enrichment_tables

    def make_path_dict(self, condition, max_sw, min_sw=0.0, path='all', dc_df=None, genes=None, norm=None):
        """ Creates a dictionary mapping paths to normalized gene counts

        Arguments:
            condition     : string key for the data set to use (i.e. KO or WT)
            max_sw  : maximum segment width, multiplied to the normalized flows, show be less than 1
            min_sw  : minimum segment width, added to compensate for "0" flows
            path    : dot-separated string specifying a certain path signature used to filter the flow dictionary

        Returns:
            Nested dictionary where first key gives the step, i.e. x coordinate, and the second coordinate is in the form
            (starting level, exiting level), i.e. y coordinates, giving the value (normalized number of genes, display
            boolean where 1 = show, 0 = hide).

        """
        if dc_df is None:
            # Get appropriate data set from cluster dictionary.
            dc_df = np.cumsum(self.cluster_dict[condition], axis=1)

            # Remove values that are all zero
            dc_df = dc_df[(dc_df.T != 0).any()]

        if genes is not None:
            dc_df = dc_df.loc[genes]

            # Drop NaNs
            dc_df = dc_df[np.isfinite(dc_df.T).all()]

        dc_array = dc_df.values

        array_min = np.min(dc_array)
        array_max = np.max(dc_array)

        # Normalization factor for flow (divide by sum of genes and multiply by maximum line width.
        if norm is None:
            norm = float(len(dc_array)) / max_sw
        else:
            norm = float(norm)/max_sw

        # Create dictionary mapping step to list of (level in, level out) to normalized flow.
        flow_dict = {step: {key: (max((value / norm), min_sw), (1 if path == 'all' else 0)) for key, value in
                            Counter(zip(dc_array[:, step], dc_array[:, step + 1])).iteritems()}
                     for step in range(0, dc_array.shape[1] - 1)}
        # Adjust flow_dictionary to mark specific segments as not displayed.
        if path != 'all':
            path_nav = [float(x) for x in path.split('.')]
            y = 0

            # Traverse path and set segments on path to display.
            for i in range(0, len(path_nav)):
                try:
                    y_next = y + path_nav[i]
                    flow_dict[i][(y, y_next)] = (flow_dict[i][(y, y_next)][0], 1)
                    y = y_next
                except:
                    pass
        return flow_dict, array_min, array_max, norm

    def plot_nodes(self, ax, flow_dict, node_width, x_coords):
        """ Plots nodes of flow dictionary

        Arguments:
            ax          : figure axis
            flow_dict   : dictionary mapping flows to x and y, generated by make_path_dict method
            node_width  : width of node rectangle
            x_coords    : list of x coordinates for plot

        Returns:
            Array of Polygon node objects

        """
        # X coordinates must be sorted otherswise nodes won't be assigned to the proper location
        x_coords.sort()

        # Iterate through each entry in the flow dictionary.
        nodes = {}
        for step, seg_dict in flow_dict.iteritems():
            levels = set([seg[0] for seg in seg_dict.keys()])
            if step == len(x_coords)-2:
                levels = levels.union(set([seg[1] for seg in seg_dict.keys()]))

            for level in levels:
                for i, h in self.calc_height(flow_dict, seg_dict, step, level).iteritems():
                    offset = self.calc_offset(flow_dict, nodes, step + i, level, h)
                    nodes[(step + i, level)] = patches.Polygon(
                        self.make_node_points(x_coords[step + i], level + offset, h, node_width),
                        fc='k', edgecolor='none')
                    ax.add_patch(nodes[(step+i, level)])

        return nodes

    def calc_height(self, flow_dict, seg_dict, x, y):
        """ Gets node heights for given step and level

        Arguments:
            flow_dict   : dictionary mapping step to seg_dicts
            seg_dict    : dictionary mapping (starting level, ending level) to segment width
            x           : step location of node
            y           : level location of node

        Returns:
            List of one (for step < 3) or two (accounts for leaf nodes) heights.

        """

        # Calculate the appropriate height based on the flow out of the current step (x) and level (y).
        heights = {}
        if x == 0:
            heights[0] = np.sum([np.abs(x) for (x, y) in seg_dict.values()])
        else:
            h_in = np.sum([np.abs(flow[0]) for seg, flow in flow_dict[x - 1].iteritems() if seg[1] == y])
            h_out = np.sum([np.abs(flow[0]) for seg, flow in flow_dict[x].iteritems() if seg[0] == y])
            heights[0] = max(h_in, h_out)

            # Old method
            # heights[0] = np.sum([flow[0] for seg, flow in flow_dict[x - 1].iteritems() if seg[1] == y])

            # Additional heights for ending nodes.
            if x == 3:
                heights[1] = np.sum([np.abs(flow[0]) for seg, flow in flow_dict[x].iteritems() if seg[1] == y])

        return heights

    def calc_offset(self, flow_dict, nodes, x, y, height):
        """ Gets y axis adjustment to center rectangles

        Arguments:
            flow_dict   : dictionary mapping step to seg_dicts
            nodes       : list of Polygon node objects
            x           : step location of node
            y           : level location of node
            height      : node height

        Returns:
            Y axis offset to be added to y coordinate.

        """
        if x > 1:
            try:
                prev_node = nodes[(x - 1, y)]  # get node horizontally to the left, if it exists
                y1 = prev_node.xy[0][1]  # get lower y coordinate of previous node
                y1_flow = flow_dict[x - 1][(y, y - 1)][0]  # flow leaving down from previous node
                y2 = y - height / 2  # get lower y coordinate of current node
                y2_flow = np.sum([np.abs(flow[0]) for seg, flow in flow_dict[x - 1].iteritems()
                                  if seg[0] < y and seg[1] == y])  # flow entering up from previous node
                return (y1 + y1_flow) - (y2 + y2_flow)
            except:
                return 0
        else:
            return 0

    def make_node_points(self, x, y, height, width):
        """ Calculates vertices of rectangle

        Arguments:
            x       : center x coordinate of rectangle
            y       : center y coordinate of rectangle
            height  : height of rectangle
            width   : width of rectangle

        Returns:
            Numpy array containing four sets of [x,y] coordinates.

        """

        # Since rectangle is centered on x and y, divide width and height by 2.
        w = width / 2.0
        h = height / 2.0

        # Set of four points, starting with lower left corner.
        return np.array([[x - w, y - h], [x + w, y - h], [x + w, y + h], [x - w, y + h]])

    def plot_polys(self, ax, flow_dict, nodes, flow_color, flow_alpha, dir):
        """ Plots the path polygons

        Arguments:
            ax          : figure axis
            flow_dict   : dictionary mapping flows to x and y, generated by make_path_dict method
            nodes       : list of Polygon node objects
            flow_color  : string representing polygon color
            flow_alpha  : float representing polygon fill alpha (between 0 and 1)
            dir         : string specifying the direction of the polygon (up, down, or over)
            up          : optional list of Polygon up objects (required for plotting 'over' rectangles
            down        : optional list of Polygon down objects (required for plotting 'over' rectangles
            node_width  : width of node rectangle
            x_coords    : list of x coordinates for plot

        Returns:
            Array of Polygon objects

        """

        # Flow polygon parameters (first level, second level, left reference point, right reference point, width direction).
        poly_dict = {'up': (0, 1, 2, 0, 1), 'down': (1, 0, 1, 3, -1)}
        if type(flow_color) is list:
            display = {0: 'none', 1: flow_color[0], -1: flow_color[1]}
        else:
            display = {0: 'none', 1: flow_color}
        polys = {}

        for step, seg_dict in flow_dict.iteritems():
            for seg, flow in seg_dict.iteritems():
                level = seg[0]
                mass = np.abs(flow[0])

                point_set = [0, 0, 0, 0]

                if dir == 'over' and seg[0] == seg[1] and self.up_patches.__len__() != 0 \
                        and self.down_patches.__len__() != 0:  # Plot rectangles
                    if self.down_patches.__contains__((step, level)):
                        point_set[0] = self.down_patches[(step, level)].xy[2]
                    else:
                        point_set[0] = nodes[(step, level)].xy[1]

                    if self.up_patches.__contains__((step, level - 1)):
                        point_set[1] = self.up_patches[(step, level - 1)].xy[0]
                    else:
                        point_set[1] = nodes[(step + 1, level)].xy[0].copy()
                        point_set[1][1] = point_set[0][1]

                    point_set[3] = np.array([point_set[0][0], point_set[0][1] + mass])
                    point_set[2] = np.array([point_set[1][0], point_set[1][1] + mass])

                    polys[(step, level)] = patches.Polygon(np.array(point_set))
                    ax.add_patch(patches.Polygon(np.array(point_set), fc=display[flow[1]],
                                                 alpha=flow_alpha, edgecolor='none'))
                elif dir != 'over' and seg[poly_dict[dir][0]] < seg[poly_dict[dir][1]]:  # Plot up or down polygons
                    try:
                        # Get reference points from appropriate node objects.
                        start_node = nodes[(step, level)]
                        end_node = nodes[(step+1, seg[1])]
                        point_set[3] = nodes[(step, level)].xy[poly_dict[dir][2]]  # left reference point
                        point_set[1] = nodes[(step + 1, seg[1])].xy[poly_dict[dir][3]]  # right reference point

                        # Adjust references values up/down in y, keep x the same.
                        point_set[2] = np.array(
                            [point_set[3][0], point_set[3][1] - (mass) * poly_dict[dir][4]])  # adjusted left
                        point_set[0] = np.array(
                            [point_set[1][0], point_set[1][1] + (mass) * poly_dict[dir][4]])  # adjusted right

                        # Save just the endpoints as a polygon to return, actual plot uses curves.
                        polys[(step, level)] = patches.Polygon(np.array(point_set))
                        ax.add_patch(patches.PathPatch(self.make_curve_path(point_set), fc=display[flow[1]],
                                                       alpha=flow_alpha, edgecolor='none'))
                    except:
                        pass

        return polys

    def plot_subpath(self, ax, path, condition, norm, min_sw):
        poly_dict = {0: self.horizontal_patches, -1: self.down_patches, 1: self.up_patches}
        point_dict = {0: (0, 1, 2, 3), -1: (3, 0, 1, 2), 1: (2, 1, 0, 3)}   # Dictionary to match array indices

        # Split the path so it's easier to work with
        path = path.split('.')

        # Remove genes that are always zero
        dc_df = self.cluster_dict[condition][(self.cluster_dict[condition].T != 0).any()]
        cum_path = np.cumsum(np.array(path).astype(int))
        path_len = len(path)
        max_path_len = len(dc_df.columns)
        if path_len < max_path_len:
            more_combos = make_binary_combos(['0', '1', '-1'], max_path_len - path_len)

        growing_path = binary_to_growing_path(range(max_path_len), dc_df, '.')
        genes_in_path = len(growing_path[growing_path.iloc[:, path_len-1] == '.'.join(path)])
        genes_in_prepath = len(growing_path[growing_path.iloc[:, path_len-2] == '.'.join(path[:-1])])
        for idx in range(1, path_len):
            step = idx-1
            start_level = cum_path[step]
            direction = cum_path[idx] - start_level
            background_poly = poly_dict[direction][(step, start_level)]
            if idx < path_len-1:
                width = genes_in_prepath/norm
                color = 'b'
            else:
                width = genes_in_path/norm
                color = 'g'
            width = max(width, min_sw)
            point_set = background_poly.xy[:-1]
            background_width = np.abs(point_set[point_dict[direction][3]][1]-point_set[point_dict[direction][0]][1])
            point_set[point_dict[direction][0]][1] += (background_width-width)/2
            point_set[point_dict[direction][1]][1] += (background_width-width)/2
            point_set[point_dict[direction][2]][1] -= (background_width-width)/2
            point_set[point_dict[direction][3]][1] -= (background_width-width)/2
            ax.add_patch(patches.PathPatch(self.make_curve_path(point_set), fc=color, edgecolor='none'))

    #todo: def get_background_patch()
    #todo: def make_future_path()

    def make_curve_path(self, point_set):
        """ Creates a curved path given endpoints

        Arguments:
            point_set   : set of four endpoints
        Returns:
            Path object containing two cubic curves

        """

        # Anchor points for curve are located halfway between the two x coordinates.
        offset = abs(point_set[1][0] - point_set[2][0]) / 2

        # Verticies of the path (last point is not drawn but needed for the CLOSEPOLY command.
        verts = [point_set[2], (point_set[2][0] + offset, point_set[2][1]), (point_set[1][0] - offset, point_set[1][1]),
                 point_set[1],
                 point_set[0], (point_set[3][0] + offset, point_set[0][1]), (point_set[0][0] - offset, point_set[3][1]),
                 point_set[3],
                 (0, 0)]

        # Path codes specifying start, end, and curves.
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CLOSEPOLY]

        return Path(verts, codes)

    def plot_flows(self, ax, sets, colors, alphas, paths, max_sw=1, min_sw=0.01, node_width=None, x_coords=None,
                   uniform=False, path_df=None, genes=None, norm=None):
        """
        Plots a Sankey-like flow figure

        :param ax: axes
            matplotlib axes object where plotting will take place
        :param sets: list
            list of data set keys
        :param colors: list
            list of corresponding flow colors
        :param alphas: list
            list of float for alpha ranging from 0 (transparent) to 1 (opaque)
        :param paths: list
            list of strings defining which paths to plot (either 'all' or string path, e.g. '01-111')
        :param max_sw: float
            maximum segment width, should be < 1
        :param min_sw: float
            minimum segment
        :param node_width: float
            width of nodes
        :param x_coords: list
            list of x coordinates for plot
        :return:
        """

        # Set axes values
        # ax.set_axis_bgcolor('0.75')
        y_min, y_max = -1.1, 1.1
        ax.set_ylim([y_min, y_max])

        if x_coords is None or uniform == True:
            x_coords = range(len(self.times))
            ax.set_xlim([np.min(x_coords), np.max(x_coords)])
            ax.set_xticks(x_coords)
            ax.set_xticklabels(np.sort(self.times))
        else:
            ax.set_xticks(self.times)

        # Scale node width
        if node_width is None:
            node_width = 0.01*(ax.get_xlim()[1]-ax.get_xlim()[0])

        if path_df is not None:
            flow_dict, path_min, path_max, norm = self.make_path_dict('NA', max_sw, min_sw, dc_df=path_df, norm=norm)  # create flow dictionary
            y_min, y_max = min(path_min - 0.1, y_min), max(path_max + 0.1, y_max)
            ax.set_ylim([y_min, y_max])
            nodes = self.plot_nodes(ax, flow_dict, node_width, x_coords)  # plot nodes
            colors = colors[0]
            self.up_patches = self.plot_polys(ax, flow_dict, nodes, colors, 1, dir='up')  # plot up polygons
            self.down_patches = self.plot_polys(ax, flow_dict, nodes, colors, 1, dir='down')  # plot down polygons
            self.horizontal_patches = self.plot_polys(ax, flow_dict, nodes, colors, 1, dir='over')  # plot rectangles

            # Resize y axis if necessary
            y_min, y_max = min(path_min - 0.1, y_min), max(path_max + 0.1, y_max)
            ax.set_ylim([y_min, y_max])
            return

        for s, c, a, p in zip(sets, colors, alphas, paths):
            if p != 'all' and p != 'diff':
                self.plot_subpath(ax, p, s, norm, min_sw)
                continue

            if s != 'diff':
                flow_dict, path_min, path_max, norm = self.make_path_dict(s, max_sw, min_sw, path=p, genes=genes, norm=norm)  # create flow dictionary

            else:
                wt_flow_dict, wt_min, wt_max, norm = self.make_path_dict('WT', max_sw, min_sw, genes=genes, norm=norm,
                                                                       path='all')  # create flow dictionary norm=norm,
                ko_flow_dict, ko_min, ko_max, norm = self.make_path_dict('KO', max_sw, min_sw, genes=genes,
                                                                       path='all')  # create flow dictionary

                path_max = max(wt_max, ko_max)
                path_min = min(wt_min, ko_min)
                diff_flow = {}
                for step in wt_flow_dict.keys():
                    diff_flow[step] = {}
                    ko_segs = set(ko_flow_dict[step].keys())
                    wt_segs = set(wt_flow_dict[step].keys())
                    diff_segs = list(ko_segs.union(wt_segs))
                    for seg in diff_segs:
                        try:
                            wt_mass = wt_flow_dict[step][seg][0]
                        except:
                            wt_mass = 0

                        try:
                            ko_mass = ko_flow_dict[step][seg][0]
                        except:
                            ko_mass = 0
                        mass_diff = wt_mass-ko_mass
                        diff_flow[step][seg] = (np.abs(mass_diff), np.sign(mass_diff))

            # Resize y axis if necessary
            y_min, y_max = min(path_min-0.1, y_min), max(path_max+0.1, y_max)
            ax.set_ylim([y_min, y_max])
            
            if s != 'diff':
                nodes = self.plot_nodes(ax, flow_dict, node_width, x_coords)  # plot nodes
                self.up_patches = self.plot_polys(ax, flow_dict, nodes, c, a, dir='up')  # plot up polygons
                self.down_patches = self.plot_polys(ax, flow_dict, nodes, c, a, dir='down')  # plot down polygons
                self.horizontal_patches = self.plot_polys(ax, flow_dict, nodes, c, a, dir='over')  # plot rectangles
            else:
                nodes = self.plot_nodes(ax, diff_flow, node_width, x_coords)  # plot nodes
                self.up_patches = self.plot_polys(ax, diff_flow, nodes, colors, a, dir='up')  # plot up polygons
                self.down_patches = self.plot_polys(ax, diff_flow, nodes, colors, a, dir='down')  # plot down polygons
                self.horizontal_patches = self.plot_polys(ax, diff_flow, nodes, colors, a, dir='over')  # plot rectangles