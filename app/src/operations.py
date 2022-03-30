import logging
import pandas as pd
from classes import LastModification
from database import Database
from datetime import datetime
import random
import networkx as nx
from networkx.classes.graph import Graph

from logic import compare_centralities, solve_eigenvalue_problem
from visualization import draw_iteration_result


def solve(graph: Graph, tensor_fn, alpha, p, num_iter):
    result = solve_eigenvalue_problem(
        graph, tensor_fn, alpha, p, num_iter)
    print(result)


def compare(graph: Graph, alpha, p, num_iter):
    table, centrality_names = compare_centralities(graph, [
        "binary", "random_walk", "clustering_coefficient", "local_closure"], alpha, p, num_iter)

    df = pd.DataFrame(table, index=pd.Index(centrality_names),
                      columns=pd.Index(centrality_names))
    df.columns.name = "p_value \ tau"
    logging.info("\n" + df.to_string())


def comparison(graph, alpha, num_iter):
    db = Database()
    p = 0
    for i in range(11):
        table, centrality_names = compare_centralities(graph, [
            "binary", "random_walk", "clustering_coefficient", "local_closure"], i*0.1, p, num_iter)
        db.insert_comparison("karate", i*0.1, args.p, table[0][1], table[0][2], table[0][3],
                             table[0][4], table[1][2], table[1][3], table[1][4], table[2][3], table[2][4], table[3][4])
    db.conn.commit()


def find_similar_centralities(graph: Graph, alpha, p, num_iter, mode):
    iteration = 0
    date_time_str = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")

    last_graph = None
    best_tau_sum = -1000

    last_modifications = []
    while True:
        # calculating centrality correlations
        table, centrality_names = compare_centralities(graph, [
            "binary", "random_walk", "clustering_coefficient", "local_closure"], alpha, p, num_iter)

        tau_sum = 0
        for i in range(len(table)):
            for j in range(i+1, len(table[i])):
                tau_sum += table[i][j]

        if tau_sum > best_tau_sum:
            best_tau_sum = tau_sum
            draw_iteration_result(
                graph.copy(), f"app/results/{date_time_str}", iteration, tau_sum, last_modifications, True)
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
                c = random.randint(0, n-1)
                d = random.randint(0, n-1)
                if c != d and ((c != a and d != b) or (c != b and d != a)) and not graph.has_edge(c, d):
                    new_graph.add_edge(c, d)
                    last_modifications.append(LastModification(c, d, True))
                    break

        last_graph = graph
        graph = new_graph

        iteration += 1
