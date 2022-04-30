from datetime import datetime
from networkx.classes.graph import Graph
from itertools import count
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import numpy as np
import subprocess

from classes import LastModification


def print_with_labels(graph: Graph, vector, file_name):
    t1_attributes = {}
    for node in graph.nodes:
        t1_attributes[node] = vector[node-1]
    nx.draw(graph, with_labels=True, labels=t1_attributes)
    # nx.draw_circular(graph, with_labels=True, labels=t1_attributes)
    plt.draw()
    plt.savefig(f"{file_name}.png")
    plt.clf()


# https://stackoverflow.com/a/28916094
def colormap(graph: Graph, attributes, file_name: str):
    groups = set(attributes)
    mapping = dict(zip(sorted(groups), count()))
    nodes = graph.nodes()
    colors = [mapping[attributes[n]] for n in nodes]

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = nx.circular_layout(graph)
    ec = nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors,
                                node_size=100, cmap=plt.cm.jet)
    # TODO possible upgrade: node_size=[v * 100 for v in nodes.values()
    plt.colorbar(nc)
    plt.axis('off')
    plt.savefig(f"{file_name}.png")
    plt.clf()


date_time_str = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
path = f"app/results/{date_time_str}"
if not os.path.exists(path):
    os.makedirs(path)


def draw_iteration_result(graph: Graph, iteration: str, value, last_modifications: list[LastModification], success: bool, tensor_fn_names):

    if len(last_modifications) != 0:
        edge_color_list = []
        for last_modification in last_modifications:
            if (success and not last_modification.is_add) or not success:
                graph.add_edge(last_modification.u,
                               last_modification.v)
        for e in graph.edges():
            modified = False
            for last_modification in last_modifications:
                if (e[0] == last_modification.u and e[1] == last_modification.v) or (e[1] == last_modification.u and e[0] == last_modification.v):
                    modified = True
                    if last_modification.is_add:
                        edge_color_list.append("limegreen")
                    else:
                        edge_color_list.append("red")
                    break
            if not modified:
                edge_color_list.append("black")

        node_color_list = []
        for n in graph.nodes():
            modified = False
            for last_modification in last_modifications:
                if n == last_modification.u or n == last_modification.v:
                    modified = True
                    if last_modification.is_add:
                        node_color_list.append("green")
                    else:
                        node_color_list.append("darkred")
                    break
            if not modified:
                node_color_list.append("tab:blue")

    else:
        edge_color_list = None
        node_color_list = None

    plt.title(f"{', '.join(tensor_fn_names)}\n{iteration}. iteration, {value}",
              color=("g" if success else "r"))
    nx.draw_circular(graph, edge_color=edge_color_list,
                     node_color=node_color_list)
    plt.savefig(f"{path}/{'__'.join(tensor_fn_names)}__{iteration}.png")
    plt.clf()


def create_gif():
    subprocess.run(
        f"cd {path} && convert -delay 50 -loop 0 `ls -v *.png` progress.gif", shell=True)


def draw_scatterplot(degrees_x, degrees_y):
    fig, ax = plt.subplots()
    ax.scatter(degrees_x, degrees_y)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("original")
    ax.set_ylabel("result")

    fig.suptitle("degree distribution")

    fig.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(f"{path}/scatterplot.png")


def draw_histograms(degrees_1, degrees_2):
    max_degree = max(max(degrees_1), max(degrees_2))
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.hist(degrees_1, range(1, max_degree))
    ax2.hist(degrees_2, range(1, max_degree))
    ax1.set_title("original")
    ax2.set_title("result")
    fig.suptitle("degree distribution")
    fig.tight_layout()
    fig.savefig(f"{path}/histograms.png")
