from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dl_scratch.common.optimizer import *

sns.set_style('whitegrid')
colors = ['#de3838', '#007bc3', '#ffd12a']
markers = ['o', 'x', ',']


def f(x: float, y: float) -> float:
    return x**2 / 20.0 + y**2


def df(x: float, y: float) -> Tuple[float, float]:
    return x / 10.0, 2.0*y


init_pos : Tuple[float, float] = (-7.0, 2.0)
params : Dict[str, float] = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads : Dict[str, float] = {}
grads['x'], grads['y'] = 0, 0


optimizers : Dict[str, Optimizer] = {}
optimizers['SGD'] = SGD(lr=0.95)
optimizers['Momentum'] = Momentum(lr=0.1)
optimizers['AdaGrad'] = AdaGrad(lr=1.5)
optimizers['Adam'] = Adam(lr=0.3)

idx : int = 1

for key in optimizers:
    optimizer : Optimizer = optimizers[key]
    x_history : List[float] = []
    y_history : List[float] = []
    params['x'], params['y'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # plot 
    plt.subplot(2, 2, idx)
    idx += 1
    plt.contour(X, Y, Z)
    plt.plot(x_history, y_history, 'o-', color=colors[0], alpha=0.8)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()