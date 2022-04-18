import logging
import pandas as pd
from utils import get_tensor_util_by_name
from classes import LastModification
from database import Database
import random
import networkx as nx
from networkx.classes.graph import Graph
import matplotlib.pyplot as plt

from logic import compare_centralities, solve_eigenvalue_problem
from visualization import create_gif, draw_histograms, draw_iteration_result, draw_scatterplot


def solve(graph: Graph, tensor_fn_name, alpha, p, num_iter):
    tensor_fn = get_tensor_util_by_name(tensor_fn_name)
    result = solve_eigenvalue_problem(
        graph, tensor_fn, alpha, p, num_iter)
    print(result)


def compare(graph: Graph, tensor_fn_names, alpha, p, num_iter):
    table = compare_centralities(graph, tensor_fn_names, alpha, p, num_iter)

    df = pd.DataFrame(table, index=pd.Index(tensor_fn_names),
                      columns=pd.Index(tensor_fn_names))
    df.columns.name = "p_value \ tau"
    logging.info("\n" + df.to_string())


def comparison(graph, tensor_fn_names, alpha, p, num_iter):
    db = Database()
    p = 0
    for i in range(11):
        table = compare_centralities(
            graph, tensor_fn_names, i*0.1, p, num_iter)
        db.insert_comparison("karate", i*0.1, p, table[0][1], table[0][2], table[0][3],
                             table[0][4], table[1][2], table[1][3], table[1][4], table[2][3], table[2][4], table[3][4])
    db.conn.commit()


def find_similar_centralities(graph: Graph, tensor_fns, alpha, p, num_iter, mode, treshold):
    iteration = 0

    last_graph = None
    best_tau_sum = -1000

    last_modifications = []
    while True:
        iteration += 1
        # calculating centrality correlations
        table = compare_centralities(
            graph, tensor_fns, alpha, p, num_iter)

        tau_sum = 0
        for i in range(len(table)):
            for j in range(i+1, len(table[i])):
                tau_sum += table[i][j]

        if tau_sum > best_tau_sum:
            best_tau_sum = tau_sum
            draw_iteration_result(
                graph.copy(), iteration, tau_sum, last_modifications, True, tensor_fns)

            if tau_sum > treshold:
                return graph
        else:
            graph = last_graph

        logging.info(f"tau_sum: {tau_sum}, best: {best_tau_sum}")

        n = len(graph.nodes)

        new_graph = graph.copy()
        last_modifications = []

        if mode == 1:
            complete_graph = graph.size() == n*(n-1)/2

            if complete_graph or (len(graph.edges) > 0 and bool(random.getrandbits(1))):
                u, v = list(graph.edges)[random.randint(
                    0, len(graph.edges)-1)]
                new_graph.remove_edge(u, v)
                last_modifications.append(LastModification(u, v, False))
            else:
                while True:
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v and not graph.has_edge(u, v):
                        new_graph.add_edge(u, v)
                        last_modifications.append(LastModification(u, v, True))
                        break
        elif mode == 2:
            while True:
                a, b = list(graph.edges)[random.randint(
                    0, len(graph.edges)-1)]
                removed_edge_graph = new_graph.copy()
                removed_edge_graph.remove_edge(a, b)
                if nx.is_connected(removed_edge_graph):
                    new_graph = removed_edge_graph
                    break

            last_modifications.append(LastModification(a, b, False))

            while True:
                c = list(graph.nodes)[random.randint(0, n-1)]
                d = list(graph.nodes)[random.randint(0, n-1)]
                if c != d and ((c != a and d != b) or (c != b and d != a)) and not graph.has_edge(c, d):
                    new_graph.add_edge(c, d)
                    last_modifications.append(LastModification(c, d, True))
                    break

        last_graph = graph
        graph = new_graph


def find_similar_pair(graph, tensor, alpha, p, num_iter, treshold):
    original_degrees = list(map(lambda x: x[1], graph.degree))
    for i in range(len(tensor) - 1):
        logging.debug(f"pair: {tensor[i]} {tensor[i+1]}")
        graph = find_similar_centralities(
            graph, tensor[i:i+2], alpha, p, num_iter,  2, treshold)
    result_degrees = list(map(lambda x: x[1], graph.degree))
    create_gif()
    draw_scatterplot(original_degrees, result_degrees)
    draw_histograms(original_degrees, result_degrees)
