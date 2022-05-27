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
import pandas as pd

from IPython.display import clear_output
from tqdm import tqdm

NQUBIT = 3 # For current testing
NEVT = 500

def data():
  # class label, lepton 1 pT, lepton 1 eta, lepton 1 phi, lepton 2 pT, lepton 2 eta, lepton 2 phi, missing energy magnitude, missing energy phi, MET_rel, axial MET, M_R, M_TR_2, R, MT2, S_R, M_Delta_R, dPhi_r_b, cos(theta_r1)
  df = pd.read_csv("Files/HIGGS_1K.csv",names=('isSignal',
      'lep1_pt','lep1_eta','lep1_phi',
      'lep2_pt','lep2_eta','lep2_phi',
      'miss_ene','miss_phi','MET_rel',
      'axial_MET','M_R','M_TR_2','R',
      'MT2','S_R',
      'M_Delta_R','dPhi_r_b','cos_theta_r1'))
  feature_dim = NQUBIT   # dimension of each data point
  if feature_dim == 3:
      SelectedFeatures = ['lep1_pt', 'lep2_pt', 'miss_ene']
  elif feature_dim == 5:
      SelectedFeatures = ['lep1_pt','lep2_pt','miss_ene','M_TR_2','M_Delta_R']
  elif feature_dim == 7:
      SelectedFeatures = ['lep1_pt','lep1_eta','lep2_pt','lep2_eta','miss_ene','M_TR_2','M_Delta_R']

  #jobn = JOBN
  training_size = NEVT
  testing_size = NEVT
  #shots = 1024
  #uin_depth = NDEPTH_UIN
  #uvar_depth = NDEPTH_UVAR
  #niter = NITER
  #backend_name = 'BACKENDNAME'
  #option = 'OPTION'
  #random_seed = 10598+1010*uin_depth+101*uvar_depth+jobn

  # Train Test split
  df_sig = df.loc[df.isSignal==1, SelectedFeatures]
  df_bkg = df.loc[df.isSignal==0, SelectedFeatures]

  df_sig_training = df_sig.values[:training_size]
  df_bkg_training = df_bkg.values[:training_size]
  df_sig_test = df_sig.values[training_size:training_size+testing_size]
  df_bkg_test = df_bkg.values[training_size:training_size+testing_size]
  training_input = {'1':df_sig_training, '0':df_bkg_training}
  test_input = {'1':df_sig_test, '0':df_bkg_test}


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


# https://qml.baidu.com/tutorials/
# machine-learning/quantum-kernel-methods.html
if __name__ == '__main__' :
  # call the function 
  # directly
  data()
  test()
  example()
