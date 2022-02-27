import argparse
from ast import arg
from typing import Callable, Dict
import networkx as nx
from networkx.classes.graph import Graph
import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

from tensor import get_binary_triangle_tensor, get_random_walk_triangle_tensor, get_clustering_coefficient_triangle_tensor, get_local_closure_triangle_tensor
from visualization import colormap
from utils import mx

logging.basicConfig(level=logging.DEBUG,  # filename=".log", filemode='a',
                    format="%(asctime)s - %(levelname)s - %(message)s")

pd.options.display.float_format = '{:.8f}'.format


def solve_eigenvalue_problem(graph: Graph, get_tensor_fn: Callable[[Graph], np.ndarray], alpha: float, p: float, num_iterations: int):
    logging.debug(
        f"Solving eigenvalue problem for: {graph.name}, {get_tensor_fn.__name__}, {alpha}, {p}, {num_iterations}")
    A = nx.adjacency_matrix(graph)
    t1 = get_tensor_fn(graph)
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_iterations):
        b_k1 = mx(A, t1, b_k, alpha, p)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k


def create_eigenvalue_centrality_images(graph: Graph, get_tensor_fn: Callable[[Graph], np.ndarray], alpha: float, p: float, num_iterations: int):
    for i in range(11):
        result = solve_eigenvalue_problem(
            graph, get_tensor_fn, alpha + i / 10, p, num_iterations)
        colormap(
            graph, result, f"../results/eigenvalue_centrality/colored_karate_Tl_{alpha+i/10}_{p}_{num_iterations}.png")


def get_tensor_util_by_name(name: str):
    if name == "binary":
        return get_binary_triangle_tensor
    elif name == "random_walk":
        return get_random_walk_triangle_tensor
    elif name == "clustering_coefficient":
        return get_clustering_coefficient_triangle_tensor
    elif name == "local_closure":
        return get_local_closure_triangle_tensor
    else:
        raise argparse.ArgumentError(f"Unknown tensor type: {name}")


def compare_centralities(graph: Graph, tensor_fn_names: list[str], alpha: float, p: float, num_iterations: int, include_degree_centrality: bool = True):
    results = []

    centrality_names = []
    if include_degree_centrality:
        centrality_names.append("degree")
        results.append(
            list(nx.algorithms.centrality.degree_centrality(graph).values()))

    centrality_names.extend(tensor_fn_names)

    for tensor_fn_name in tensor_fn_names:
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

    return table, centrality_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph', type=argparse.FileType("r"),
                        help="graph to work with")
    parser.add_argument('-t', '--tensor', type=str,
                        default="binary", help="type of the tensor to use")
    parser.add_argument('-a', '--alpha', type=float, default=0.5,
                        help="first/second order interaction ratio (0<=a<=1)")
    parser.add_argument('-p', type=float, default=0,
                        help="parameter of the T_p operator")
    parser.add_argument('-n', '--num_iter', type=int, default=10,
                        help="number of iterations in the power method")
    parser.add_argument('operation', choices=[
                        "solve", "compare"], help="what to do")

    args = parser.parse_args()
    logging.debug(f"args: {args}")

    if args.graph is None:
        graph = nx.karate_club_graph()
    else:
        raise argparse.ArgumentError(
            "Custom graph processing not implemented yet")

    tensor_fn = get_tensor_util_by_name(args.tensor)

    if args.operation == "solve":
        result = solve_eigenvalue_problem(
            graph, tensor_fn, args.alpha, args.p, args.num_iter)
    elif args.operation == "compare":
        table, centrality_names = compare_centralities(graph, [
            "binary", "random_walk", "clustering_coefficient", "local_closure"], args.alpha, args.p, args.num_iter)

        df = pd.DataFrame(table, index=pd.Index(centrality_names),
                          columns=pd.Index(centrality_names))
        df.columns.name = "p_value \ tau"
        logging.info("\n" + df.to_string())
    else:
        raise argparse.ArgumentError(f"Invalid operation: {args.operation}")
