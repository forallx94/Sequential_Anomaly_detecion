import argparse
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

from model.LSTM_Autoencoder import LSTM_Autoencoder
from model.NN_Autoencoder import NN_Autoencoder

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--x', 
                    help='x_npy data path',default='./data/Averaged_BearingTest_Dataset.npy')
parser.add_argument('--result', 
                    help='result csv path',default='./result')

args = parser.parse_args()

X_train = np.load(args.x)

result_model = []

result_model.append(LSTM_Autoencoder(X_train))
result_model.append(NN_Autoencoder(X_train))