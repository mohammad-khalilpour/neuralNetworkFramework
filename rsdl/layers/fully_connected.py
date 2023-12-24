from ..tensors import Tensor
from .init import initializer, zero_initializer


class Linear:

    def __init__(self, in_channels, out_channels, need_bias=True, mode='xavier') -> None:
        self.shape = (in_channels, out_channels)
        self.need_bias = need_bias
        self.weight = Tensor(
            data=initializer([in_channels, out_channels], mode),
            requires_grad=True
        )
        if self.need_bias:
            self.bias = Tensor(
                data=zero_initializer([out_channels]),
                requires_grad=True
            )

    def forward(self, inp: Tensor) -> Tensor:
        z = inp.__matmul__(self.weight).__add__(self.bias)
        return z
    
    def parameters(self):
        
        if self.need_bias:
            return [self.weight, self.bias]
        return [self.weight]
    
    def zero_grad(self):
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def __call__(self, inp):
        return self.forward(inp)
