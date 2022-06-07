import numpy as np
from nptyping import NDArray, Shape, Int, Float
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


from simple_convnet import SimpleConvNet
from dl_scratch.common.trainer import Trainer

sns.set_style('whitegrid')
colors = ['#de3838', '#007bc3', '#ffd12a']


# MNISTデータの読み込み
(x_train, t_train), (x_test, t_test) = mnist.load_data()
t_train : NDArray[Shape['60000, 10'], Int] = to_categorical(t_train)
t_test : NDArray[Shape['60000, 10'], Int] = to_categorical(t_test)
x_train : NDArray[Shape['60000, 1, 28, 28'], Float] = (x_train/255).reshape(-1, 1, 28, 28) 
x_test : NDArray[Shape['60000, 1, 28, 28'], Float] = (x_test/255).reshape(-1 ,1, 28, 28)

# 処理に時間のかかる場合はデータを削減 
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
