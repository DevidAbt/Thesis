import argparse
from typing import Callable
import networkx as nx
from networkx.classes.graph import Graph
import numpy as np

from tensor import get_binary_triangle_tensor, get_random_walk_triangle_tensor, get_clustering_coefficient_triangle_tensor, get_local_closure_triangle_tensor
from visualization import colormap
from utils import mx


def solve_eigenvalue_problem(graph: Graph, get_tensor_fn: Callable[[Graph], np.ndarray], alpha: float, p: float, num_iterations: int):
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
    parser.add_argument('operation', choices=["solve"], help="what to do")

    args = parser.parse_args()
    print(f"args: {args}")

    if args.graph is None:
        graph = nx.karate_club_graph()
    else:
        raise argparse.ArgumentError(
            "Custom graph processing not implemented yet")

    if args.tensor == "binary":
        tensor_fn = get_binary_triangle_tensor
    elif args.tensor == "random_walk":
        tensor_fn = get_random_walk_triangle_tensor
    elif args.tensor == "clustering_coefficient":
        tensor_fn = get_clustering_coefficient_triangle_tensor
    elif args.tensor == "local_closure":
        tensor_fn = get_local_closure_triangle_tensor
    else:
        raise argparse.ArgumentError(f"Unknown tensor type: {args.tensor}")

    if args.operation == "solve":
        result = solve_eigenvalue_problem(
            graph, tensor_fn, args.alpha, args.p, args.num_iter)
        print(result)
    else:
        raise argparse.ArgumentError(f"Invalid operation: {args.operation}")
