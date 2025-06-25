import numpy as np
from sklearn.datasets import make_moons, make_blobs

"""
Source: https://github.com/ethangilmore/MicroMLP
"""

def relu(x):
    y = np.maximum(0, x)

    def backward(dy):
        dx = dy * (y > 0)
        return dx

    return y, backward


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))

    def backward(dy):
        dx = dy * y * (1 - y)
        return dx

    return y, backward


def tanh(x):
    y = np.tanh(x)

    def backward(dy):
        dx = dy * (1 - y ** 2)
        return dx

    return y, backward


def softmax(x):
    y = np.exp(x) / sum(np.exp(x))

    def backward(dy):
        J = -np.outer(y, y)
        np.fill_diagonal(J, y * (1 - y))
        return dy @ J

    return y, backward


class Parameter:
    def __init__(self, value):
        self.value = value
        self.gradient = np.zeros_like(value)

    def apply_gradient(self, step_size):
        self.value -= step_size * self.gradient
        self.gradient = np.zeros_like(self.value)


def mean_squared_error(y_pred, y_true):
    y = np.mean((y_pred - y_true) ** 2)
    dy = 2 * (y_pred - y_true) / y_pred.size
    return y, dy


def cross_entropy(y_pred, y_true):

    ## Inserted Code
    return y_pred
    ## Inserted Code

    y = -sum(y_true * np.log(y_pred))
    dy = -y_true / y_pred
    return y, dy


class Layer:
    def __init__(self, in_size: int, out_size: int, activation):
        self.w = Parameter(np.random.normal(0, 1, (out_size, in_size)))
        self.b = Parameter(np.zeros(out_size))
        self._activation_fn = activation
        self._backward = lambda: None

    def __call__(self, x: np.array) -> np.array:
        a, da_dz = self._activation_fn(self.w.value @ x + self.b.value)

        def backward(dL_da: np.array):
            dL_dz = da_dz(dL_da)
            self.w.gradient += np.outer(dL_dz, x)
            self.b.gradient += dL_dz
            return dL_dz @ self.w.value

        self._backward = backward
        return a

    def parameters(self):
        return [self.w, self.b]


class MLP:
    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x: np.array) -> np.array:
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, dy: np.array):
        for layer in reversed(self.layers):
            dy = layer._backward(dy)

    def calculate_gradients(self, x: np.array, y_true: np.array, loss):
        y_pred = self(x)
        loss, dy = loss(y_pred, y_true)
        self.backward(dy)
        return loss

    def training_step(self, xs, ys, loss, step_size):
        avg_loss = 0
        for x, y in zip(xs, ys):
            avg_loss += self.calculate_gradients(x, y, loss) / len(xs)
        for layer in self.layers:
            for p in layer.parameters():
                p.apply_gradient(step_size / len(xs))
        return avg_loss

    def train(self, xs, ys, loss, epochs, batch_size, learning_rate):
        for epoch in range(epochs):
            avg_loss = 0
            for i in range(0, len(xs), batch_size):
                avg_loss += self.training_step(xs[i:i + batch_size], ys[i:i + batch_size], loss,
                                               learning_rate) / batch_size
            print(f"Epoch {epoch}: Avg Loss {avg_loss}")