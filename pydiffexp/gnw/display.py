from io import BytesIO

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from nxpd import draw
from pydiffexp.plot import DEPlot


def get_graph(path):
    """
    Get the digraph
    :param path:
    :return:
    """
    # Read the data
    net_df = pd.read_csv(path, sep='\t', header=None)

    # Set edge colors
    net_df[net_df == '+'] = 'green'
    net_df[net_df == '-'] = 'red'
    net_df.columns = ['source', 'target', 'color']

    # Make a networkx diagram
    dg = nx.from_pandas_dataframe(net_df, source='source', target='target', create_using=nx.DiGraph(),
                                  edge_attr='color')
    return dg


def draw_net(g, dpi=300, **kwargs):
    """
    Draw the network digram
    :param g:
    :param dpi:
    :return:
    """
    kwargs.setdefault('show', 'ipynb')
    kwargs.setdefault('layout', 'neato')
    # Set dpi
    g.graph['dpi'] = dpi

    # todo: if show=True this will return a string to the tmp location of the image
    gviz = draw(g, **kwargs)

    img = mpimg.imread(BytesIO(gviz.data))

    return img


def draw_results(data: pd.DataFrame, perturb, titles, times=None, samey=True, g=None, axarr=None, **kwargs):
    """

    :param data:
    :param net:
    :param perturb:
    :param data_dir:
    :param times:
    :param axarr:
    :return:
    """
    idx = pd.IndexSlice
    data.sort_index(axis=1, inplace=True)
    draw_data = data.loc[:, idx[:, :, perturb, :]]
    if times:
        draw_data = draw_data.loc[:, idx[:, :, :, times]]

    y_max = draw_data.values.max()
    dep = DEPlot()
    nodes = draw_data.index.values
    n_axes = len(nodes)
    show_net = (g is not None)
    if show_net:
        net_img = draw_net(g, **kwargs)
        n_axes += 1

    if axarr is None:
        fig, axarr = plt.subplots(1, n_axes, figsize=(15, 5))

    for ii, ax in enumerate(axarr.flatten()):
        if ii == 0:
            ax.imshow(net_img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.axis('off')
        else:
            dep.tsplot(draw_data.loc[nodes[ii-show_net]], ax=ax, subgroup='Time', legend=False,
                       mean_line_dict={'ls': '--'})
            ax.set_ylabel('Normalized Expression')
            if samey:
                ax.set_ylim([0, y_max])
            # ax.set_xlabel('')
            ax.set_title(titles[ii-1])
            if ii > 1:
                ax.get_yaxis().set_visible(False)

            if ii > 0:
                ax.set_ylabel('')

    return axarr