import argparse
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

from model.LSTM_Autoencoder import lstm_autoencoder
from model.NN_Autoencoder import nn_autoencoder
from model.Isolation_Forest import isolation_forest
from model.One_class_SVM import one_class_svm
from model.Iso_2Auto import iso_2auto

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--x', 
                    help='x_npy data path',default='./data/Averaged_BearingTest_Dataset.npy')

args = parser.parse_args()

X_train = np.load(args.x)

result_model = []

result_model.append(isolation_forest(X_train))
result_model.append(one_class_svm(X_train))
result_model.append(iso_2auto(X_train))
result_model.append(lstm_autoencoder(X_train))
result_model.append(nn_autoencoder(X_train))