from typing import List, Dict, Optional

import numpy as np
from nptyping import NDArray, Shape, Int, Float

from dl_scratch.common.layers import *
from dl_scratch.common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01) -> None:
        # 重みの初期化
        self.params : Dict[str, NDArray] = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers : Dict[str, Layer] = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer : SoftmaxWithLoss = SoftmaxWithLoss()
        
    def predict(self, x: NDArray) -> NDArray:
        for layer in self.layers.values():
            x : NDArray = layer.forward(x)
        
        return x
        
    # x:入力データ, t:教師データ
    def loss(self, x: NDArray, t: NDArray) -> float:
        y : NDArray = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x: NDArray, t: NDArray) -> float:
        y : NDArray = self.predict(x)
        y : NDArray = np.argmax(y, axis=1)
        if t.ndim != 1: 
            t = np.argmax(t, axis=1)
        
        accuracy : float = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x: NDArray, t: NDArray) -> Dict[str, NDArray]:
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout : NDArray = self.lastLayer.backward(dout)
        
        layers : List[Layer] = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads : Dict[str, NDArray] = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
