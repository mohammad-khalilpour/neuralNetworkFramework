import time

import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import Adam, SGD, Momentum
from rsdl.losses.loss_functions import categorical_cross_entropy, mean_squared_error
from rsdl.activations.activation_functions import relu, softmax, sigmoid
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train_set = datasets.MNIST(
    download=True,
    root='Data',
    train=True,
    transform=ToTensor(),
)

test_set = datasets.MNIST(
    download=True,
    root='Data',
    train=False,
    transform=ToTensor(),
)
batch_size = 100
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

fc1 = Linear(784, 100)
fc2 = Linear(100, 10)

optimizer = Adam(layers=[fc2, fc1], learning_rate=0.01)


for epoch in range(50):

    epoch_loss = 0.0
    batch = 0

    for images, labels in train_loader:
        batch += 1

        flattened_images = images.view([images.size(0), -1])
        numpy_images = flattened_images.numpy()
        inputs = Tensor(numpy_images)

        # forward part
        z1 = fc1.forward(inputs)
        fc1_output = relu(z1)
        z2 = fc2.forward(fc1_output)
        predicted = softmax(z2)

        actual = Tensor(np.zeros((len(labels), 10)))
        actual.data[np.arange(len(labels)), labels] = 1

        loss = categorical_cross_entropy(predicted, actual)

        if batch == 3:
            count = 0
            correct_count = 0
            for i, pred in enumerate(predicted.data):
                pred_class = 0
                pred_chance = 0
                for j, pred_c in enumerate(pred):
                    if pred_c > pred_chance:
                        pred_class = j
                        pred_chance = pred_c
                # print(f'{pred_class}:{labels[i]}')
                count += 1
                if labels[i] == pred_class:
                    correct_count += 1
            print(correct_count / count)
            print(loss.data)

        loss.backward()
        epoch_loss += loss.data

        optimizer.step()
        optimizer.zero_grad()

