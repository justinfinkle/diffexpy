import sys, inspect
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable.colorbrewer as cbrewer

from pydiffexp import DEResults, DEAnalysis

# Set plot defaults
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = 'Arial'
axes = {'labelsize': 28,
        'titlesize': 28}
mpl.rc('axes', **axes)
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 24

_colors = cbrewer.qualitative.Dark2_8.mpl_colors
_paired = cbrewer.qualitative.Paired_9.mpl_colors


class DiffExpPlot(object):
    def __init__(self, dea):
        self.palette = _colors
        self.dea = dea                              # type: DEAnalysis

    def volcano_plot(self, df: pd.DataFrame, p_value: float = 0.05, fc=2, x_colname='logFC', y_colname='-log10p',
                     cutoff_lines=True, top_n=None, top_by='-log10p', show_labels=False):

        # Get rid of NaN data
        df = df.dropna()

        # Convert cutoffs to logspace
        log2_fc = np.log2(fc)
        log10_pval = -np.log10(p_value)

        # Split data into above and below cutoff dataframes
        sig = df[(df[y_colname] >= log10_pval) & (np.abs(df[x_colname]) >= log2_fc)]
        insig = df[~(df[y_colname] >= log10_pval) | ~(np.abs(df[x_colname]) >= log2_fc)]

        # Get maximum values for formatting latter
        max_y = np.max(sig[y_colname])
        max_x = np.ceil(np.max(np.abs(sig[x_colname])))

        fig, ax = plt.subplots(figsize=(10, 10))

        # Split top data points if requested
        if top_n:

            # Find points to highlight
            sort = set()
            if isinstance(top_by, list):
                for col in top_by:
                    sort = sort.union(set(sig.index[np.argsort(np.abs(sig[col]))[::-1]][:top_n].values))
            elif isinstance(top_by, str):
                sort = sort.union(set(sig.index[np.argsort(np.abs(sig[top_by]))[::-1]][:top_n].values))
            else:
                raise ValueError('top_by must be a string or list of values found in the DataFrame used for the plot')

            top_sig = sig.loc[sort]
            sig = sig.drop(sort)
            ax.plot(top_sig[x_colname], top_sig[y_colname], 'o', c=_colors[0], ms=10, zorder=2, label='Top Genes')

            if show_labels:
                fs = mpl.rcParams['legend.fontsize']
                for row in top_sig.iterrows():
                    ax.annotate(row[0], xy=(row[1][x_colname], row[1][y_colname]), fontsize=fs, style='italic')

        # Make plot
        ax.plot(sig[x_colname], sig[y_colname], 'o', c=_colors[2], ms=10, zorder=1, label='Diff Exp')
        ax.plot(insig[x_colname], insig[y_colname], 'o', c=_colors[-1], ms=10, zorder=0, mew=0, label='')

        # Adjust axes
        ax.set_xlim([-max_x, max_x])
        ax.set_ylim([0, max_y])

        # Add cutoff lines
        if cutoff_lines:
            color = _colors[1]
            # P value line

            ax.plot([-max_x, max_x], [log10_pval, log10_pval], '--', c=color, lw=3, label='Threshold')

            # log fold change lines
            ax.plot([-log2_fc, -log2_fc], [0, max_y], '--', c=color, lw=3)
            ax.plot([log2_fc, log2_fc], [0, max_y], '--', c=color, lw=3)

        ax.legend(loc='best', numpoints=1)

        # Adjust labels
        ax.tick_params(axis='both', which='major')
        ax.set_xlabel(r'$log_2(\frac{KO}{WT})$')
        ax.set_ylabel(r'$-log_{10}$(corrected p-value)')
        plt.show()

    def add_ts(self, ax, data, name, subgroup='time', mean_line_dict=None, fill_dict=None, ci=0.83):
        gene = data.name
        data = data.reset_index()
        grouped_data = data.groupby(subgroup)

        # Get plotting statistics. Rows are: group, mean, SE, and Tstat
        grouped_stats = np.array(
            [[g, np.mean(data[gene]), stats.sem(data[gene]), stats.t.ppf(1 - (1 - ci) / 2, df=len(data) - 1)]
             for g, data in grouped_data]).T

        if mean_line_dict is None:
            mean_line_dict = dict()
        if fill_dict is None:
            fill_dict = dict()
        mean_defaults = dict(ls='-', marker='s', lw=2, mew=0, label=(name + " mean"), ms=10, zorder=0)
        mean_kwargs = dict(mean_defaults, **mean_line_dict)
        mean_line, = ax.plot(grouped_stats[0], grouped_stats[1], **mean_kwargs)
        mean_color = mean_line.get_color()
        jitter_x = data[subgroup]  # +(np.random.normal(0, 1, len(data)))
        ax.plot(jitter_x, data[gene], '.', color=mean_color, ms=15, label='', alpha=0.5)

        fill_defaults = dict(lw=0, facecolor=mean_color, alpha=0.2, label=(name + (' {:d}%CI'.format(int(ci * 100)))))
        fill_kwargs = dict(fill_defaults, **fill_dict)
        ax.fill_between(grouped_stats[0], grouped_stats[1] - grouped_stats[2] * grouped_stats[3],
                        grouped_stats[1] + grouped_stats[2] * grouped_stats[3], **fill_kwargs)

    def tsplot(self, df, supergroup='condition', subgroup='time'):
        gene = df.name
        supers = sorted(list(set(df.index.get_level_values(supergroup))))
        print(supers)
        fig, ax = plt.subplots()
        ax.set_prop_cycle(cycler('color', _colors))
        for sup in supers:
            sup_data = df.loc[sup]
            self.add_ts(ax, sup_data, sup, subgroup=subgroup)
        # ax.set_xlim([np.min(grouped_stats[0]), np.max(grouped_stats[0])])
        ax.legend(loc='best', numpoints=1)
        # ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: '{:,.0f}'.format(x)))
        ax.set_xlabel(subgroup.title())
        ax.set_ylabel('Expression')
        ax.set_title(gene)
        plt.tight_layout()

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
            norm = float(norm) / max_sw

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
            if step == len(x_coords) - 2:
                levels = levels.union(set([seg[1] for seg in seg_dict.keys()]))

            for level in levels:
                for i, h in self.calc_height(flow_dict, seg_dict, step, level).iteritems():
                    offset = self.calc_offset(flow_dict, nodes, step + i, level, h)
                    nodes[(step + i, level)] = patches.Polygon(
                        self.make_node_points(x_coords[step + i], level + offset, h, node_width),
                        fc='k', edgecolor='none')
                    ax.add_patch(nodes[(step + i, level)])

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
                        end_node = nodes[(step + 1, seg[1])]
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
        point_dict = {0: (0, 1, 2, 3), -1: (3, 0, 1, 2), 1: (2, 1, 0, 3)}  # Dictionary to match array indices

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
        genes_in_path = len(growing_path[growing_path.iloc[:, path_len - 1] == '.'.join(path)])
        genes_in_prepath = len(growing_path[growing_path.iloc[:, path_len - 2] == '.'.join(path[:-1])])
        for idx in range(1, path_len):
            step = idx - 1
            start_level = cum_path[step]
            direction = cum_path[idx] - start_level
            background_poly = poly_dict[direction][(step, start_level)]
            if idx < path_len - 1:
                width = genes_in_prepath / norm
                color = 'b'
            else:
                width = genes_in_path / norm
                color = 'g'
            width = max(width, min_sw)
            point_set = background_poly.xy[:-1]
            background_width = np.abs(point_set[point_dict[direction][3]][1] - point_set[point_dict[direction][0]][1])
            point_set[point_dict[direction][0]][1] += (background_width - width) / 2
            point_set[point_dict[direction][1]][1] += (background_width - width) / 2
            point_set[point_dict[direction][2]][1] -= (background_width - width) / 2
            point_set[point_dict[direction][3]][1] -= (background_width - width) / 2
            ax.add_patch(patches.PathPatch(self.make_curve_path(point_set), fc=color, edgecolor='none'))

            # todo: def get_background_patch()
            # todo: def make_future_path()


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
            node_width = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])

        if path_df is not None:
            flow_dict, path_min, path_max, norm = self.make_path_dict('NA', max_sw, min_sw, dc_df=path_df,
                                                                      norm=norm)  # create flow dictionary
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
                flow_dict, path_min, path_max, norm = self.make_path_dict(s, max_sw, min_sw, path=p, genes=genes,
                                                                          norm=norm)  # create flow dictionary

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
                        mass_diff = wt_mass - ko_mass
                        diff_flow[step][seg] = (np.abs(mass_diff), np.sign(mass_diff))

            # Resize y axis if necessary
            y_min, y_max = min(path_min - 0.1, y_min), max(path_max + 0.1, y_max)
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