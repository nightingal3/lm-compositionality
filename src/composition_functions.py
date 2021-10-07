import numpy as np
from typing import Callable

add = lambda arr: np.sum(arr, axis=0)
element_wise_mult = lambda x, y: np.multiply(x, y)

def weighted_vector_add(alpha: float, beta: float) -> Callable:
    def add(x, y):
        return alpha * x + beta * y
    return add