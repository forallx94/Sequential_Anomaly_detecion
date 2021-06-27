import argparse
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

from model import lstm_autoencoder, nn_autoencoder, isolation_forest, one_class_svm, iso_2auto


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--x', 
                        help='x_npy data path',default='./data/Averaged_BearingTest_Dataset.npy')

    args = parser.parse_args()

    X_train = np.load(args.x)

    result_model = []
    result_model.append(nn_autoencoder(X_train))
    result_model.append(lstm_autoencoder(X_train))
    result_model.append(isolation_forest(X_train))
    result_model.append(one_class_svm(X_train))
    result_model.append(iso_2auto(X_train))
    