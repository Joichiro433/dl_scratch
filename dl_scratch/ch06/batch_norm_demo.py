import imp
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from dl_scratch.common.multi_layer_net_extend import MultiLayerNetExtend
from dl_scratch.common.optimizer import SGD, Adam

sns.set_style('whitegrid')
colors = ['#de3838', '#007bc3', '#ffd12a']
markers = ['o', 'x', ',']

# MNISTデータの読み込み
(x_train, t_train), (x_test, t_test) = mnist.load_data()
t_train : NDArray[Shape['60000, 10'], Int] = to_categorical(t_train)
t_test : NDArray[Shape['60000, 10'], Int] = to_categorical(t_test)
x_train : NDArray[Shape['60000, 784'], Float] = x_train.reshape(60000, 784) / 255
x_test : NDArray[Shape['60000, 784'], Float] = x_test.reshape(10000, 784) / 255

# 学習データを削減
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std: Union[str, float]) -> Tuple[List[float], List[float]]:
    bn_network : MultiLayerNetExtend = MultiLayerNetExtend(
        input_size=784, 
        hidden_size_list=[100, 100, 100, 100, 100], 
        output_size=10, 
        weight_init_std=weight_init_std, 
        use_batchnorm=True)
    network : MultiLayerNetExtend = MultiLayerNetExtend(
        input_size=784, 
        hidden_size_list=[100, 100, 100, 100, 100], 
        output_size=10,
        weight_init_std=weight_init_std)
    optimizer : SGD = SGD(lr=learning_rate)
    
    train_acc_list : List[float] = []
    bn_train_acc_list : List[float] = []
    
    iter_per_epoch : int = int(max(train_size / batch_size, 1))
    epoch_cnt : int = 0
    
    for i in range(1000000000):
        batch_mask : NDArray[Shape['100'], Int] = np.random.choice(train_size, batch_size)
        x_batch : NDArray[Shape['100, 784'], Float] = x_train[batch_mask]
        t_batch : NDArray[Shape['100, 10'], Float] = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads : Dict[str, NDArray] = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            train_acc : float = network.accuracy(x_train, t_train)
            bn_train_acc : float = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# グラフの描画
weight_scale_list : NDArray[Shape['16'], Float] = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4,4,i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()