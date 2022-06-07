from turtle import forward
from typing import Dict, Tuple, Optional
from abc import ABCMeta, abstractmethod

import numpy as np
from nptyping import NDArray

from dl_scratch.common.functions import *
from dl_scratch.common.util import im2col, col2im


class Layer(metaclass=ABCMeta):
    @abstractmethod
    def forward(self):
        raise NotImplementedError()
    
    @abstractmethod
    def backward(self):
        raise NotImplementedError()


class Relu(Layer):
    def __init__(self):
        self.mask : Optional[NDArray[bool]] = None

    def forward(self, x: NDArray) -> NDArray:
        self.mask = (x <= 0)
        out : NDArray = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: NDArray) -> NDArray:
        dout[self.mask] = 0
        dx : NDArray = dout
        return dx


class Sigmoid(Layer):
    def __init__(self):
        self.out : Optional[NDArray] = None

    def forward(self, x: NDArray) -> NDArray:
        out : NDArray = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout: NDArray) -> NDArray:
        dx : NDArray = dout * (1.0 - self.out) * self.out
        return dx


class Affine(Layer):
    def __init__(self, W: NDArray, b: NDArray):
        self.W : NDArray = W
        self.b : NDArray = b
        
        self.x : Optional[NDArray] = None
        self.original_x_shape : Optional[Tuple(int, ...)] = None
        # 重み, バイアスパラメータの微分
        self.dW : Optional[NDArray] = None
        self.db : Optional[NDArray] = None

    def forward(self, x: NDArray) -> NDArray:
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out : NDArray = self.x @ self.W + self.b

        return out

    def backward(self, dout: NDArray) -> NDArray:
        dx : NDArray = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class SoftmaxWithLoss(Layer):
    def __init__(self):
        self.loss : Optional[float] = None
        self.y : Optional[NDArray] = None # softmaxの出力
        self.t : Optional[NDArray] = None # 教師データ

    def forward(self, x: NDArray, t: NDArray) -> float:
        self.t : NDArray = t
        self.y : NDArray = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1) -> NDArray:
        batch_size : int = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx : NDArray = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization(Layer):
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma: NDArray, beta: NDArray, momentum: float = 0.9, running_mean: Optional[NDArray] = None, running_var: Optional[NDArray] = None):
        self.gamma : NDArray = gamma
        self.beta : NDArray = beta
        self.momentum : float = momentum
        self.input_shape : Optional[Tuple[int, ...]] = None # Conv層の場合は4次元、全結合層の場合は2次元  s

        # テスト時に使用する平均と分散
        self.running_mean : Optional[NDArray] = running_mean
        self.running_var : Optional[NDArray] = running_var  
        
        # backward時に使用する中間データ
        self.batch_size : Optional[int] = None
        self.xc : Optional[NDArray] = None
        self.std : Optional[NDArray] = None
        self.dgamma : Optional[NDArray] = None
        self.dbeta : Optional[NDArray] = None

    def forward(self, x: NDArray, train_flg: bool = True) -> NDArray:
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out : NDArray = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x: NDArray, train_flg: bool) -> NDArray:
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu : NDArray = x.mean(axis=0)
            xc : NDArray = x - mu
            var : NDArray = np.mean(xc**2, axis=0)
            std : NDArray = np.sqrt(var + 10e-7)
            xn : NDArray = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out : NDArray = self.gamma * xn + self.beta 
        return out

    def backward(self, dout: NDArray) -> NDArray:
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout : NDArray = dout.reshape(N, -1)

        dx : NDArray = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout: NDArray) -> NDArray:
        dbeta : NDArray = dout.sum(axis=0)
        dgamma : NDArray = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution(Layer):
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling(Layer):
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
