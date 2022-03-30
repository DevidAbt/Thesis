import networkx as nx
from networkx.classes.graph import Graph
import numpy as np


def get_binary_triangle_tensor(graph: Graph):
    A = nx.adjacency_matrix(graph)
    length = A.shape[0]
    tensor = np.ndarray((length, length, length))

    for i in range(length):
        for j in range(length):
            for k in range(length):
                value = 1 if i != j and i != k and j != k \
                    and A[i, j] > 0 and A[i, k] > 0 and A[j, k] > 0 else 0
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
                    and A[i, j] > 0 and A[i, k] > 0 and A[j, k] > 0 else 0
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
                    and A[i, j] > 0 and A[i, k] > 0 and A[j, k] > 0 else 0
                tensor[i][j][k] = value

    return tensor


def get_local_closure_triangle_tensor(graph: Graph):
    A = nx.adjacency_matrix(graph)
    length = A.shape[0]
    adjacency_matrix = A.toarray()

    one_vector = np.ones(length)
    d = np.matmul(adjacency_matrix, one_vector)
    w = np.subtract(np.matmul(adjacency_matrix, d), d)

    tensor = np.ndarray((length, length, length))

    for i in range(length):
        for j in range(length):
            for k in range(length):
                value = 1 / w[i] if i != j and i != k and j != k \
                    and A[i, j] > 0 and A[i, k] > 0 and A[j, k] > 0 else 0
                tensor[i][j][k] = value

    return tensor
