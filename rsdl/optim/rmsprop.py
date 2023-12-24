from rsdl.optim import Optimizer
from rsdl import Tensor
import numpy as np


class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.001, decay=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon

        for layer in self.layers:
            layer.squared_grad_w = np.zeros_like(layer.weight)
            if layer.need_bias:
                layer.squared_grad_b = np.zeros_like(layer.bias)

    def step(self):
        for layer in self.layers:

            layer.squared_grad_w = self.decay * layer.squared_grad_w + (1 - self.decay) * layer.weight.grad**2
            layer.weight = layer.weight - self.learning_rate * layer.weight.grad / (np.sqrt(layer.squared_grad_w) + self.epsilon)

            if layer.need_bias:
                layer.squared_grad_b = self.decay * layer.squared_grad_b + (1 - self.decay) * layer.bias.grad ** 2
                layer.bias = layer.bias - self.learning_rate * layer.bias.grad / (np.sqrt(layer.squared_grad_b) + self.epsilon)
