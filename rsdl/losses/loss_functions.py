import numpy as np
from rsdl.tensors import Tensor


def mean_squared_error(preds: Tensor, actual: Tensor) -> Tensor:
    subs_sum = preds.__sub__(actual)
    subs_sum = subs_sum.__pow__(2)
    subs_sum = subs_sum.sum()
    subs_sum.data /= preds.shape[0]
    return subs_sum


def categorical_cross_entropy(preds: Tensor, actual: Tensor):
    ce = actual.__mul__(preds.log()).sum().__neg__()
    return ce



