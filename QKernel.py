# Author: Shiwen AN 
# Date: 2022/05/29 
# Purpose: Trying to 
# catch the possibility 
# using quantum kernel method

from ast import Global
import time
# from tkinter.font import names
import matplotlib
import numpy as np
import pandas as pd
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


# Global variable for manual updates of the progress bar
N = 1
G = 2
# The QKE circuit simulated by paddle quantm
def q_kernel_estimator(x1, x2):
    
    # Transform data vectors into tensors
    x1 = paddle.to_tensor(x1)
    x2 = paddle.to_tensor(x2)
    
    # Create the circuit
    cir = paddle_quantum.ansatz.Sequential()
    
    # Add the encoding circuit for the first data vector
    cir.append(IQPEncoding(qubits_idx=[[0,1]], feature=x1))
    init_state = paddle_quantum.state.zero_state(G)
    state = cir[0](state=init_state)    
    
    # Add inverse of the encoding circuit for the second data vector
    cir.append(IQPEncoding(qubits_idx=[[0,1]], feature=x2))
    fin_state = cir[1](state=state,invert=True).data
    
    # Update the progress bar
    global N
    pbar.update(100/N)
    
    # Return the probability of measuring 0...0 
    return (fin_state[0].conj() * fin_state[0]).real().numpy()[0]

# Define a kernel matrix function, for which the input should be two list of vectors
# This is needed to customize the SVM kernel
def q_kernel_matrix(X1, X2):
    return np.array([[q_kernel_estimator(x1, x2) for x2 in X2] for x1 in X1])

# progress bar displaying the progress
bar_format_string = '{l_bar}{bar}|[{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]'
pbar = tqdm(total=100, bar_format=bar_format_string)

# Visualize the decision function, boundary, and margins of +- 0.2
def QKernel(X_train, X_test, y_train,y_test):
    pbar = tqdm( 100,
        desc='Training and predicting with QKE-SVM',
        bar_format=bar_format_string
    )
    sig_N = len(X_train)**2+ len(X_train)**2+len(X_train)*len(X_test)
    svm_qke = svm.SVC(kernel=q_kernel_matrix)
    svm_qke.fit(X_train, y_train)

    # See how the svm classifies the training and testing data
    predict_svm_qke_train = svm_qke.predict(X_train)
    predict_svm_qke_test = svm_qke.predict(X_test)

    # Calculate the accuracy
    accuracy_train = np.array(predict_svm_qke_train == y_train, dtype=int).sum()/len(y_train)
    accuracy_test = np.array(predict_svm_qke_test == y_test, dtype=int).sum()/len(y_test)

    print(accuracy_train)
    print(accuracy_test)


# Take the data and train
def data(feature_dim, size):
  df = pd.read_csv("Files/HIGGS_10K.csv",names=('isSignal',
  'lep1_pt', 'lep1_eta', 'lep1_phi', 'miss_ene', 'miss_phi', 
  'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_b_tag', 
  'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_b_tag', 
  'jet3_pt', 'jet3_eta', 'jet3_phi', 'jet3_b_tag', 
  'jet4_pt', 'jet4_eta', 'jet4_phi', 'jet4_b_tag', 
  'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'))

  # Train Test split
  if feature_dim == 3:
      SelectedFeatures = ['isSignal','lep1_pt', 'jet1_pt', 'miss_ene']
  elif feature_dim == 4:
      SelectedFeatures = ['isSignal','lep1_pt', 'jet1_pt','jet1_eta','miss_ene']     
  elif feature_dim == 5:
      SelectedFeatures = ['isSignal','lep1_pt','jet1_pt','miss_ene','jet2_pt','m_jj']
  elif feature_dim == 7:
      SelectedFeatures = ['isSignal','lep1_pt','lep1_eta','jet1_pt','jet1_eta','miss_ene','m_jj','m_bb']

  df_sig = df[SelectedFeatures]
  df_sig = df_sig.values[:size]
  print("Length of the array", len(df_sig))
  inputs = df_sig[:,list(range(1,feature_dim+1))]
  global G 
  G = feature_dim 
  signals = df_sig[:,0]
  
  inputs_train, inputs_test, signals_train, signals_test = train_test_split(inputs, signals, 
                                                                random_state=529, test_size=0.2)

  print(inputs_train)
  print(signals_train)
  print("Length of the dataset is: ", len(inputs_train))

  QKernel(inputs_train, inputs_test, signals_train, signals_test)



if __name__ == '__main__' :
  # call the function 
  # directly
  data(4,1000)