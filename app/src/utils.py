import math
import numpy as np


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
