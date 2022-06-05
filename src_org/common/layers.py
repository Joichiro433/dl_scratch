from typing import Optional, Tuple

import numpy as np
from nptyping import NDArray

from common.functions import *


class Relu:
    def __init__(self) -> None:
        self.mask : NDArray[bool] = None

    def forward(self, x: NDArray) -> NDArray:
        self.mask = (x <= 0)
        out : NDArray = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: NDArray) -> NDArray:
        dout[self.mask] = 0
        dx : NDArray = dout
        return dx


class Sigmoid:
    def __init__(self) -> None:
        self.out : Optional[NDArray] = None

    def forward(self, x: NDArray) -> NDArray:
        out : NDArray = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: NDArray) -> NDArray:
        dx : NDArray = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W: NDArray, b: NDArray) -> None:
        self.W : NDArray = W
        self.b : NDArray = b
        
        self.x : Optional[NDArray] = None
        self.original_x_shape : Optional[Tuple[int, ...]] = None
        # 重み・バイアスパラメータの微分
        self.dW : Optional[NDArray] = None
        self.db : Optional[NDArray] = None

    def forward(self, x: NDArray) -> NDArray:
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out : NDArray = self.x @ self.W + self.b

        return out

    def backward(self, dout : NDArray) -> NDArray:
        dx : NDArray = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

