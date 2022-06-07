from typing import Dict

import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
colors = ['#de3838', '#007bc3', '#ffd12a']
markers = ['o', 'x', ',']


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-x))


def ReLU(x: NDArray) -> NDArray:
    return np.maximum(0, x)


def tanh(x: NDArray) -> NDArray:
    return np.tanh(x)
    

input_data : NDArray[Shape['1000, 100'], Float] = np.random.randn(1000, 100)  # 1000個のデータ
node_num : int = 100  # 各隠れ層のノード（ニューロン）の数
hidden_layer_size : int = 5  # 隠れ層が5層
activations : Dict[int, NDArray] = {}  # ここにアクティベーションの結果を格納する

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 初期値の値をいろいろ変えて実験しよう！
    w : NDArray[Shape['100, 100'], Float] = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    a : NDArray[Shape['1000, 100'], Float] = x @ w


    # 活性化関数の種類も変えて実験しよう！
    z : NDArray[Shape['1000, 100'], Float] = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# ヒストグラムを描画
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
