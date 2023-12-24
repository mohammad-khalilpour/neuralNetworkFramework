from typing import List
from ..layers.fully_connected import Linear


class Optimizer:
    def __init__(self, layers: List[Linear]):
        self.layers = layers
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
