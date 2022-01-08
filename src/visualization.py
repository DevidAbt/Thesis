from networkx.classes.graph import Graph
from itertools import count
import networkx as nx
import matplotlib.pyplot as plt


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
    plt.colorbar(nc)
    plt.axis('off')
    plt.savefig(f"{file_name}.png")
    plt.clf()
