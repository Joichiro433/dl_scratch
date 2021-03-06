import numpy as np
from nptyping import NDArray


def identity_function(x: NDArray) -> NDArray:
    return x


def step_function(x: NDArray) -> NDArray:
    return np.array(x > 0, dtype=np.int64)


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: NDArray) -> NDArray:
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x: NDArray) -> NDArray:
    return np.maximum(0, x)


def relu_grad(x: NDArray) -> NDArray:
    grad : NDArray = np.zeros_like(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x: NDArray) -> NDArray:
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y: NDArray, t: NDArray) -> float:
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y: NDArray, t: NDArray) -> float:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X: NDArray, t: NDArray) -> float:
    y = softmax(X)
    return cross_entropy_error(y, t)
