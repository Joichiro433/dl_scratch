from typing import Optional, List, Dict, Any
import sys, os
sys.path.append(os.pardir)

import numpy as np
from nptyping import NDArray
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from common.layers import *
from common.gradient import numerical_gradient


SampleSize = 0
InputSize = 0
HiddenSize = 0
OutputSize = 0

class TwoLayerNet:
    def __init__(
            self, 
            input_size: int, 
            hidden_size: int, 
            output_size: int,
            weight_init_std: float = 0.01) -> None:
        self.params : Dict[str, NDArray] = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers : Dict[str, Any] = {}
        self.layers['Affine1'] = Affine(W=self.params['W1'], b=self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(W=self.params['W2'], b=self.params['b2'])
        self.lastLayer : SoftmaxWithLoss = SoftmaxWithLoss()

    def predict(self, x: NDArray[(SampleSize, InputSize)]) -> NDArray[SampleSize, OutputSize]:
        for layer in self.layers.values():
            x : NDArray = layer.forward(x)
        return x

    def loss(self, x: NDArray, t: NDArray):
        y : NDArray[SampleSize, OutputSize] = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x: NDArray, t: NDArray) -> float:
        y : NDArray[SampleSize, OutputSize] = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy : float = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x: NDArray, t: NDArray) -> Dict[str, NDArray]:
        loss_W = lambda W: self.loss(x, t)
        
        grads : Dict[str, NDArray] = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

    def gradient(self, x: NDArray, t: NDArray) -> Dict[str, NDArray]:
        self.loss(x, t)

        dout : NDArray = self.lastLayer.backward()
        layers: List[Any] = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads : Dict[str, NDArray] = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

        
        
if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    train_images = train_images.reshape(60000, 784) / 255
    test_images = test_images.reshape(10000, 784) / 255

    network : TwoLayerNet = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    x_batch = train_images[:3]
    t_batch = train_labels[:3]

    grad_numerical : Dict[str, NDArray] = network.numerical_gradient(x_batch, t_batch)
    grad_backprop : Dict[str, NDArray] = network.gradient(x_batch, t_batch)

    for key in tqdm(grad_numerical.keys()):
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(f'{key}: {diff}')

    