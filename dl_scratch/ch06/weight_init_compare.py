from typing import List, Dict, Optional, Union

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from dl_scratch.common.util import smooth_curve
from dl_scratch.common.multi_layer_net import MultiLayerNet
from dl_scratch.common.optimizer import SGD

sns.set_style('whitegrid')
colors = ['#de3838', '#007bc3', '#ffd12a']
markers = ['o', 'x', ',']

# MNISTデータの読み込み
(x_train, t_train), (x_test, t_test) = mnist.load_data()
t_train : NDArray[Shape['60000, 10'], Int] = to_categorical(t_train)
t_test : NDArray[Shape['60000, 10'], Int] = to_categorical(t_test)
x_train : NDArray[Shape['60000, 784'], Float] = x_train.reshape(60000, 784) / 255
x_test : NDArray[Shape['60000, 784'], Float] = x_test.reshape(10000, 784) / 255

train_size : int = x_train.shape[0]
batch_size : int = 128
max_iterations : int = 2000


# 実験の設定
weight_init_types : Dict[str, Union[str, float]] = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
optimizer : SGD = SGD(lr=0.01)

networks : Dict[str, MultiLayerNet] = {}
train_loss : Dict[str, List[float]] = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(
        input_size=784, 
        hidden_size_list=[100, 100, 100, 100],
        output_size=10, 
        weight_init_std=weight_type)
    train_loss[key] = []


# 訓練の開始
for i in tqdm(range(max_iterations)):
    batch_mask : NDArray[Shape['128'], Int] = np.random.choice(train_size, batch_size)
    x_batch : NDArray[Shape['128, 784'], Float] = x_train[batch_mask]
    t_batch : NDArray[Shape['128, 10'], Int] = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads : Dict[str, NDArray] = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
    
        loss : float = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# グラフの描画
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()