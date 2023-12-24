import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import Adam, SGD, Momentum
from rsdl.losses.loss_functions import mean_squared_error

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

fc = Linear(3, 1)

optimizer = Adam(layers=[fc])

batch_size = 5

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        predicted = fc.forward(inputs)

        actual = y[start:end]
        actual.data = actual.data.reshape(batch_size, 1)
        loss = mean_squared_error(predicted, actual)
        
        loss.backward()
        print(loss.data)

        epoch_loss += loss

        optimizer.step()
        fc.zero_grad()


print(fc.weight)
print(fc.bias)
