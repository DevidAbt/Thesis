import argparse
import math
import numpy as np

from tensor import get_binary_triangle_tensor, get_random_walk_triangle_tensor, get_clustering_coefficient_triangle_tensor, get_local_closure_triangle_tensor


def mu(a, b, p):
    if p == 0:
        return math.sqrt(a*b)

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
