import argparse
import networkx as nx
import pandas as pd
import logging
from operations import find_similar_centralities2

from operations import compare, comparison, find_similar_centralities, solve
from utils import get_tensor_util_by_name, mx

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

pd.options.display.float_format = '{:.8f}'.format


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
                        "solve", "compare", "comparison", "find-similar-centralities", "find-similar-centralities2"], help="what to do")

    args = parser.parse_args()
    logging.debug(f"args: {args}")

    if args.graph is None:
        graph = nx.karate_club_graph()
    else:
        graph = nx.read_weighted_edgelist(args.graph)

    tensor_fn = get_tensor_util_by_name(args.tensor)

    if args.operation == "solve":
        solve(graph, tensor_fn, args.alpha, args.p, args.num_iter)
    elif args.operation == "compare":
        compare(graph, args.alpha, args.p, args.num_iter)
    elif args.operation == "comparison":
        comparison(graph, args.alpha, args.num_iter)
    elif args.operation == "find-similar-centralities":
        find_similar_centralities(graph, args.alpha, args.p, args.num_iter, 1)
    elif args.operation == "find-similar-centralities2":
        find_similar_centralities(graph, args.alpha, args.p, args.num_iter, 2)
    else:
        raise argparse.ArgumentError(f"Invalid operation: {args.operation}")
