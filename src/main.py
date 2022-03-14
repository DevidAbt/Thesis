import argparse
from datetime import datetime
import random
from typing import Callable
import networkx as nx
from networkx.classes.graph import Graph
import numpy as np
import pandas as pd
import scipy.stats as stats
import logging
from database import Database

from tensor import get_binary_triangle_tensor, get_random_walk_triangle_tensor, get_clustering_coefficient_triangle_tensor, get_local_closure_triangle_tensor
from classes import LastModification
from visualization import colormap, draw_iteration_result, print_with_labels
from utils import mx

logging.basicConfig(level=logging.DEBUG,  # filename=".log", filemode='a',
                    format="%(asctime)s - %(levelname)s - %(message)s")

pd.options.display.float_format = '{:.8f}'.format


def solve_eigenvalue_problem(graph: Graph, get_tensor_fn: Callable[[Graph], np.ndarray], alpha: float, p: float, num_iterations: int):
    logging.debug(
        f"Solving eigenvalue problem for: {graph.name}, {get_tensor_fn.__name__}, {alpha}, {p}, {num_iterations}")
    A = nx.adjacency_matrix(graph)
    t1 = get_tensor_fn(graph)
    # b_k = np.random.rand(A.shape[1])
    b_k = np.ones(A.shape[1])
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
                        "solve", "compare", "comparison", "find-similar-centralities"], help="what to do")

    args = parser.parse_args()
    logging.debug(f"args: {args}")

    if args.graph is None:
        graph = nx.karate_club_graph()
    else:
        graph = nx.read_weighted_edgelist(args.graph)

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
    elif args.operation == "comparison":
        db = Database()
        p = 0
        for i in range(11):
            table, centrality_names = compare_centralities(graph, [
                "binary", "random_walk", "clustering_coefficient", "local_closure"], i*0.1, args.p, args.num_iter)
            db.insert_comparison("karate", i*0.1, args.p, table[0][1], table[0][2], table[0][3],
                                 table[0][4], table[1][2], table[1][3], table[1][4], table[2][3], table[2][4], table[3][4])
        db.conn.commit()
    elif args.operation == "find-similar-centralities":
        iteration = 0
        date_time_str = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

        last_graph = None
        best_tau_sum = -1000

        last_modification = None
        while True:
            # calculating centrality correlations
            table, centrality_names = compare_centralities(graph, [
                "binary", "random_walk", "clustering_coefficient", "local_closure"], args.alpha, args.p, args.num_iter)

            tau_sum = 0
            for i in range(len(table)):
                for j in range(i+1, len(table[i])):
                    tau_sum += table[i][j]

            if tau_sum > best_tau_sum:
                best_tau_sum = tau_sum
                draw_iteration_result(
                    graph.copy(), f"results/{date_time_str}", iteration, tau_sum, last_modification, True)
            else:
                graph = last_graph

            logging.info(f"tau_sum: {tau_sum}, best: {best_tau_sum}")

            n = len(graph.nodes)
            complete_graph = graph.size() == n*(n-1)/2

            new_graph = graph.copy()

            edge_removed = False
            if complete_graph or (len(graph.edges) > 0 and bool(random.getrandbits(1))):
                u, v = list(graph.edges)[random.randint(
                    0, len(graph.edges)-1)]
                new_graph.remove_edge(u, v)
                last_modification = LastModification(u, v, False)
                edge_removed = True
            else:
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and not graph.has_edge(u, v):
                        new_graph.add_edge(u, v)
                        last_modification = LastModification(u, v, True)
                        break

            last_graph = graph
            graph = new_graph

            iteration += 1

    else:
        raise argparse.ArgumentError(f"Invalid operation: {args.operation}")
