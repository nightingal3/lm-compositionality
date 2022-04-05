import numpy as np
from typing import Callable
import torch

add = lambda arr: np.sum(arr, axis=0)
mult = lambda arr: np.prod(arr, axis=0)
torch_add = lambda arr: torch.sum(arr, axis=0)
torch_mult = lambda arr: torch.prod(arr, axis=0)
b_torch_add = lambda arr: torch.sum(arr, axis=1)
b_torch_mult = lambda arr: torch.prod(arr, axis=1)
b_w1 = lambda arr: arr[:, 0, :]
b_w2 = lambda arr: arr[:, 1, :]

def weighted_vector_add(alpha: float, beta: float) -> Callable:
    def add(x, y):
        return alpha * x + beta * y
    return add