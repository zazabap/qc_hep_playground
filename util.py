# Author: Shiwen An 
# Date: 2022/05/26
# Purpose: Build up 
# functions for using quantum machine learning
# in the field of high energy physic

import time
import matplotlib
import numpy as np
import paddle
from numpy import pi as PI
from matplotlib import pyplot as plt

from paddle import matmul, transpose
from paddle_quantum.ansatz import Circuit
from paddle_quantum.gate import IQPEncoding
import paddle_quantum

import sklearn
from sklearn import svm
from sklearn.datasets import fetch_openml, make_moons, make_circles
from sklearn.model_selection import train_test_split

from IPython.display import clear_output
from tqdm import tqdm


class QKM:
  def __init__(self, estimator ):
    self.estimator = estimator


def test():
  qkm_model = QKM(10)
  print(qkm_model.estimator)


def example():
  # Generate data set
  X_train, y_train = make_circles(10, noise=0.05, factor=0.2, random_state=0)
  X_test, y_test = make_circles(10, noise=0.05, factor=0.2, random_state=1024)
  
  # Visualize respectively the training and testing set
  fig, ax = plt.subplots(1, 2, figsize=[10, 4])
  ax[0].scatter(X_train[:,0], X_train[:,1], 
                marker='o', c = matplotlib.cm.coolwarm(np.array(y_train, dtype=np.float32)))
  ax[0].set_title('Train')
  ax[1].set_title('Test')
  ax[1].scatter(X_test[:,0], X_test[:,1], marker='v', c = matplotlib.cm.coolwarm(np.array(y_test, dtype=np.float32)))
  print("Let's first see our training and testing set:")
  bar_format_string = '{l_bar}{bar}|[{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
  pbar = tqdm(total=100, bar_format=bar_format_string)
  pbar.close()
  clear_output()



# https://qml.baidu.com/tutorials/
# machine-learning/quantum-kernel-methods.html
if __name__ == '__main__' :
  # call the function 
  # directly
  test()
  example()
