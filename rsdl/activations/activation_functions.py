from rsdl import Tensor, Dependency
import numpy as np


def sigmoid(t: Tensor) -> Tensor:
    t_neg = t.__neg__()
    exp = t_neg.exp()
    exp = exp.__add__(Tensor(np.array(1.0)))

    return exp.__pow__(-1)


def tanh(t: Tensor) -> Tensor:
    exp = t.exp()
    neg_exp = t.__neg__().exp()
    tanh_tensor = exp.__sub__(neg_exp).__mul__(exp.__add__(neg_exp).__pow__(-1))

    return tanh_tensor


def softmax(t: Tensor) -> Tensor:
    exp = t.exp()
    sum_exp = exp.__matmul__(np.ones((exp.shape[-1], 1)))
    softmax_tensor = exp.__mul__(sum_exp.__pow__(-1))

    return softmax_tensor


def relu(t: Tensor) -> Tensor:

    data = np.maximum(0, t.data)

    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(data > 0, 1, 0)
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def leaky_relu(t: Tensor, leak=0.05) -> Tensor:
    
    data = np.maximum(leak * t.data, t.data)
    
    req_grad = t.requires_grad
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(data > 0, 1, leak)
        
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
