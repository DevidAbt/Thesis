from networkx.classes.graph import Graph
from typing import Callable
import numpy as np
import logging
import networkx as nx
from visualization import colormap
from utils import get_tensor_util_by_name, mx
import scipy.stats as stats


def create_eigenvalue_centrality_images(graph: Graph, get_tensor_fn: Callable[[Graph], np.ndarray], alpha: float, p: float, num_iterations: int):
    for i in range(11):
        result = solve_eigenvalue_problem(
            graph, get_tensor_fn, alpha + i / 10, p, num_iterations)
        colormap(
            graph, result, f"../results/eigenvalue_centrality/colored_karate_Tl_{alpha+i/10}_{p}_{num_iterations}.png")


def solve_eigenvalue_problem(graph: Graph, get_tensor_fn: Callable[[Graph], np.ndarray], alpha: float, p: float, num_iterations: int):
    logging.debug(
        f"Solving eigenvalue problem for: {graph.name}, {get_tensor_fn.__name__}, {alpha}, {p}, {num_iterations}")
    A = nx.adjacency_matrix(graph)
    t1 = get_tensor_fn(graph)
    b_k = np.ones(A.shape[1])
    for _ in range(num_iterations):
        b_k1 = mx(A, t1, b_k, alpha, p)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k


def compare_centralities(graph: Graph, tensor_fn_names: list[str], alpha: float, p: float, num_iterations: int):
    results = []

    for tensor_fn_name in tensor_fn_names:
        if tensor_fn_name == "degree":
            results.append(
                list(nx.algorithms.centrality.degree_centrality(graph).values()))
        else:
            tensor_fn = get_tensor_util_by_name(tensor_fn_name)

            result = solve_eigenvalue_problem(
                graph, tensor_fn, alpha, p, num_iterations)

            results.append(result)

    n = len(results)
    table = []
    for i in range(n):
        table.append([])
        for j in range(n):
            if i == j:
                table[i].append(1)
            else:
                table[i].append(-100)

    for i in range(n):
        for j in range(i+1, n):
            tau, p_value = stats.kendalltau(results[i], results[j])
            table[i][j] = tau
            table[j][i] = p_value

    return table
