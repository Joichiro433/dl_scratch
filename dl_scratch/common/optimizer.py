from typing import Dict, Optional
from abc import ABCMeta, abstractmethod

import numpy as np
from nptyping import NDArray


class Optimizer(metaclass=ABCMeta):
    @abstractmethod
    def update(self, params: Dict[str, NDArray], grads: Dict[str, NDArray]) -> None:
        raise NotImplementedError()


class SGD(Optimizer):
    """確率的勾配降下法（Stochastic Gradient Descent）"""
    def __init__(self, lr : float = 0.01) -> None:
        self.lr : float = lr
        
    def update(self, params: Dict[str, NDArray], grads: Dict[str, NDArray]) -> None:
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum(Optimizer):
    """Momentum SGD"""
    def __init__(self, lr : float = 0.01, momentum : float = 0.9) -> None:
        self.lr : float = lr
        self.momentum : float = momentum
        self.v : Optional[Dict[str, NDArray]] = None
        
    def update(self, params: Dict[str, NDArray], grads: Dict[str, NDArray]) -> None:
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


class Nesterov(Optimizer):
    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""
    def __init__(self, lr : float = 0.01, momentum : float = 0.9) -> None:
        self.lr : float = lr
        self.momentum : float = momentum
        self.v : Optional[Dict[str, NDArray]] = None
        
    def update(self, params: Dict[str, NDArray], grads: Dict[str, NDArray]) -> None:
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]


class AdaGrad(Optimizer):
    """AdaGrad"""
    def __init__(self, lr : float = 0.01) -> None:
        self.lr : float = lr
        self.h : Optional[Dict[str, NDArray]] = None
        
    def update(self, params: Dict[str, NDArray], grads: Dict[str, NDArray]) -> None:
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop(Optimizer):
    """RMSprop"""
    def __init__(self, lr : float = 0.01, decay_rate : float = 0.99) -> None:
        self.lr : float = lr
        self.decay_rate : float = decay_rate
        self.h : Optional[Dict[str, NDArray]] = None
        
    def update(self, params: Dict[str, NDArray], grads: Dict[str, NDArray]) -> None:
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam(Optimizer):
    """Adam (http://arxiv.org/abs/1412.6980v8)"""
    def __init__(self, lr : float = 0.001, beta1 : float = 0.9, beta2 : float = 0.999) -> None:
        self.lr : float = lr
        self.beta1 : float = beta1
        self.beta2 : float = beta2
        self.iter : int = 0
        self.m : Optional[Dict[str, NDArray]] = None
        self.v : Optional[Dict[str, NDArray]] = None
        
    def update(self, params: Dict[str, NDArray], grads: Dict[str, NDArray]) -> None:
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
