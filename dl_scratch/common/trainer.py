from typing import List, Dict, Tuple, Optional, Union

import numpy as np
from nptyping import NDArray, Shape, Int, Float
from dl_scratch.common.optimizer import *
from dl_scratch.common.multi_layer_net_extend import MultiLayerNetExtend

class Trainer:
    """ニューラルネットの訓練を行うクラス"""
    def __init__(
            self, 
            network: MultiLayerNetExtend, 
            x_train: NDArray, 
            t_train: NDArray, 
            x_test: NDArray, 
            t_test: NDArray,
            epochs: int = 20, 
            mini_batch_size: int = 100,
            optimizer: str = 'SGD', 
            optimizer_param: Dict[str, Union[str, float]] = {'lr':0.01}, 
            evaluate_sample_num_per_epoch: Optional[int] = None, 
            verbose: bool = True) -> None:
        self.network : MultiLayerNetExtend = network
        self.verbose : bool = verbose
        self.x_train : NDArray = x_train
        self.t_train : NDArray = t_train
        self.x_test : NDArray = x_test
        self.t_test : NDArray = t_test
        self.epochs : int = epochs
        self.batch_size : int = mini_batch_size
        self.evaluate_sample_num_per_epoch : Optional[int] = evaluate_sample_num_per_epoch

        # optimizer
        optimizer_class_dict : Dict[str, Optimizer] = {
            'sgd':SGD, 
            'momentum':Momentum, 
            'nesterov':Nesterov,
            'adagrad':AdaGrad, 
            'rmsprop':RMSprop, 
            'adam':Adam
        }
        self.optimizer : Optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size : int = x_train.shape[0]
        self.iter_per_epoch : int = int(max(self.train_size / mini_batch_size, 1))
        self.max_iter : int = int(epochs * self.iter_per_epoch)
        self.current_iter : int = 0
        self.current_epoch : int = 0
        
        self.train_loss_list : List[float] = []
        self.train_acc_list : List[float] = []
        self.test_acc_list : List[float] = []

    def train_step(self) -> None:
        batch_mask : NDArray[Shape['*'], Int] = np.random.choice(self.train_size, self.batch_size)
        x_batch : NDArray[Shape['*, ...'], Float] = self.x_train[batch_mask]
        t_batch : NDArray[Shape['*, ...'], Int] = self.t_train[batch_mask]
        
        grads : Dict[str, NDArray] = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss : float = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]
                
            train_acc : float = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc : float = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self) -> None:
        for _ in range(self.max_iter):
            self.train_step()

        test_acc : float = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

