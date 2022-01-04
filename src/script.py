from typing import Callable
import matplotlib.pyplot as plt
import networkx as nx
from networkx.classes.graph import Graph
import numpy as np
from itertools import count


def get_binary_triangle_tensor(graph: Graph):
    A = nx.adjacency_matrix(graph)
    length = A.shape[0]
    tensor = np.ndarray((length, length, length))

    for i in range(length):
        for j in range(length):
            for k in range(length):
                value = 1 if i != j and i != k and j != k \
                    and A[i, j] == 1 and A[i, k] == 1 and A[j, k] == 1 else 0
                tensor[i][j][k] = value

    return tensor


def get_random_walk_triangle_tensor(graph: Graph):
    A = nx.adjacency_matrix(graph)
    length = A.shape[0]
    adjacency_matrix = A.toarray()

    tensor = np.ndarray((length, length, length))

    triangles_matrix = np.multiply(
        adjacency_matrix, np.matmul(adjacency_matrix, adjacency_matrix))

    for i in range(length):
        for j in range(length):
            for k in range(length):
                value = 1 / triangles_matrix[j][k] if i != j and i != k and j != k \
                    and A[i, j] == 1 and A[i, k] == 1 and A[j, k] == 1 else 0
                tensor[i][j][k] = value

    return tensor


def get_clustering_coefficient_triangle_tensor(graph: Graph):
    A = nx.adjacency_matrix(graph)
    length = A.shape[0]
    adjacency_matrix = A.toarray()

    one_vector = np.ones(length)
    d = np.matmul(adjacency_matrix, one_vector)

    tensor = np.ndarray((length, length, length))

    for i in range(length):
        for j in range(length):
            for k in range(length):
                value = 1 / (d[i]*(d[i]-1)) if i != j and i != k and j != k \
                    and A[i, j] == 1 and A[i, k] == 1 and A[j, k] == 1 else 0
                tensor[i][j][k] = value

    return tensor


def mu(a, b, p):
    return np.power((np.power(np.abs(a), p)+np.power(np.abs(b), p))/2, 1/p)


def tp(tensor, x, p):
    length = len(tensor)
    result = []
    for i in range(length):
        element_sum = 0
        for j in range(length):
            for k in range(length):
                element_sum += tensor[i][j][k] * mu(x[j], x[k], p)
        result.append(element_sum)
    return result


def mx(matrix, tensor, x, alpha, p):
    return np.multiply(matrix.dot(x), alpha) + np.multiply(1-alpha, tp(tensor, x, p))


def print_with_labels(graph: Graph, vector, file_name):
    t1_attributes = {}
    for node in graph.nodes:
        t1_attributes[node] = vector[node-1]
    nx.draw(G, with_labels=True, labels=t1_attributes)
    # nx.draw_circular(graph, with_labels=True, labels=t1_attributes)
    plt.draw()
    plt.savefig(f"{file_name}.png")
    plt.clf()


# def power_iteration(matrix, num_simulations: int):
#     b_k = np.random.rand(matrix.shape[1])
#     for _ in range(num_simulations):
#         b_k1 = np.dot(matrix, b_k)
#         b_k1_norm = np.linalg.norm(b_k1)
#         b_k = b_k1 / b_k1_norm
#     return b_k

def solve_eigenvalue_problem(graph: Graph, get_tensor_fn: Callable[[Graph], np.ndarray], alpha: float, p: float, num_iterations: int):
    A = nx.adjacency_matrix(graph)
    t1 = get_tensor_fn(G)
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_iterations):
        b_k1 = mx(A, t1, b_k, alpha, p)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k


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


if __name__ == "__main__":
    G = nx.karate_club_graph()

    # A = nx.adjacency_matrix(G)

    alpha = 0
    p = 2
    num_iterations = 10
    for i in range(11):
        result = solve_eigenvalue_problem(
            G, get_random_walk_triangle_tensor, alpha + i / 10, p, num_iterations)
        colormap(
            G, result, f"results/colored_karate_Tw_{alpha+i/10}_{p}_{num_iterations}.png")
