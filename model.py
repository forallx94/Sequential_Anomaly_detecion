import tensorflow as tf
from tensorflow.keras.models import  Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LSTM , RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)
RAND_SEED = 42

MODLE_PATH = './Trained models/'
GRAPH_PATH = './result/'

# define the autoencoder network model
def nn_autoencoder_model(X):
  inputs = Input(shape=(X.shape[1],))
  input_dim = X.shape[1]
  encode = Dense(input_dim//3*2, activation='relu',kernel_regularizer=regularizers.l2(0.01))(inputs)
  encode = BatchNormalization()(encode)
  encode = Dense(input_dim//3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encode)
  encode = BatchNormalization()(encode)
  encode = Dense(1, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encode)
  encode = BatchNormalization()(encode)

  decode = Dense(input_dim//3, activation='relu')(encode)
  decode = BatchNormalization()(decode)
  decode = Dense(input_dim//3*2, activation='relu')(decode)
  decode = BatchNormalization()(decode)
  output = Dense(input_dim, activation='sigmoid')(decode)
  model = Model(inputs=inputs, outputs=output)
  return model

# Autoencoder model.
def nn_autoencoder(X_train):
  train_dataset = X_train.copy()
  
  autoencoder = nn_autoencoder_model(train_dataset)

  autoencoder.compile(optimizer='adam',
             loss='mse',
             metrics=['accuracy'])

  # Train model.
  history = autoencoder.fit(train_dataset, train_dataset,
                          epochs=200,
                          batch_size=128,
                          shuffle=True,
                          validation_split=0.2,
                          callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=3),
                                        ModelCheckpoint(MODLE_PATH + 'nn_autoencoder.h5',monitor='val_loss', save_best_only=True, mode='min')]
                          )

  # plot the training losses
  fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
  ax.plot(history.history['loss'], 'b', label='Train', linewidth=2)
  ax.set_title('Model loss', fontsize=16)
  ax.set_ylabel('Loss (mae)')
  ax.set_xlabel('Epoch')
  ax.legend(loc='upper right')
  plt.savefig(GRAPH_PATH + 'NN_loss.png')

  # plot the loss distribution of the training set
  X_pred = autoencoder.predict(train_dataset)
  X_pred = pd.DataFrame(X_pred)
  scored = pd.DataFrame()
  Xtrain = train_dataset
  scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
  plt.figure(figsize=(16,9), dpi=80)
  plt.title('Loss Distribution', fontsize=16)
  sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
  plt.xlim([0.0,.5])
  plt.savefig(GRAPH_PATH  + 'NN_loss_distribution.png')

  return autoencoder


def lstm_autoencoder(X_train):
  train_dataset = np.expand_dims(X_train,axis=1)
  input_dim = train_dataset.shape[-1]

  # LSTM Autoencoder
  model = Sequential()

  # Encoder
  model.add(LSTM(input_dim//3*2, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(input_dim//3, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(1, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  # model.add(RepeatVector(10))

  # Decoder
  model.add(LSTM(input_dim//3, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(input_dim//3*2, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(TimeDistributed(Dense(input_dim)))

  model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

  # fit
  history = model.fit(train_dataset, train_dataset,
                      epochs=200, 
                      batch_size=128,
                      shuffle=True,
                      validation_split=0.2,
                      callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=3),
                                    ModelCheckpoint(MODLE_PATH + 'lstm_autoencoder.h5',monitor='val_loss', save_best_only=True, mode='min')]
                      )
  
  # plot the training losses
  fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
  ax.plot(history.history['loss'], 'b', label='Train', linewidth=2)
  ax.set_title('Model loss', fontsize=16)
  ax.set_ylabel('Loss (mae)')
  ax.set_xlabel('Epoch')
  ax.legend(loc='upper right')
  plt.savefig(GRAPH_PATH + 'LSTM_loss.png')

  # plot the loss distribution of the training set
  X_pred = model.predict(train_dataset)
  X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
  X_pred = pd.DataFrame(X_pred)

  scored = pd.DataFrame()
  Xtrain = train_dataset.reshape(train_dataset.shape[0],train_dataset.shape[2])
  scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
  plt.figure(figsize=(16,9), dpi=80)
  plt.title('Loss Distribution', fontsize=16)
  sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
  plt.xlim([0.0,.5])
  plt.savefig(GRAPH_PATH + 'LSTM_loss_distribution.png')

  return model
  

def isolation_forest(X_train):
    clf = IsolationForest(random_state = RAND_SEED)
    clf.fit(X_train)
    pred = clf.predict(X_train)

    pca = PCA(n_components=2,random_state = RAND_SEED)
    printcipalComponents = pca.fit_transform(X_train)
    principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = [1, -1]
    colors = ['g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = pred == target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component1']
                , principalDf.loc[indicesToKeep, 'principal component2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig(GRAPH_PATH + 'Isolation_Forest_PCA.png')

    return clf
  
  
def one_class_svm(X_train):
    clf = OneClassSVM(gamma='auto').fit(X_train)
    pred = clf.predict(X_train)
    pred_score = clf.score_samples(X_train)

    pca = PCA(n_components=2,random_state = RAND_SEED)
    pca.fit(X_train)
    printcipalComponents = pca.transform(X_train)
    principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = [1, -1]
    colors = ['g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = pred == target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component1']
                , principalDf.loc[indicesToKeep, 'principal component2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()

    plt.savefig(GRAPH_PATH + 'One_class_SVM_PCA.png')
    return clf


def iso_2auto(X_train):
    # 1st Autoencoder
    autoencoder1 = nn_autoencoder_model(X_train)
    autoencoder1.compile(optimizer='adam', 
                         loss='mse', 
                         metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    history1 = autoencoder1.fit(X_train, X_train,
                                epochs=200,
                                batch_size=10,
                                shuffle=True,
                                validation_split=0.2,
                                callbacks = [early_stopping,
                                              ModelCheckpoint(MODLE_PATH + 'iso_2auto_1st.h5',monitor='val_loss', save_best_only=True, mode='min')])
    print('========== First Autoencoder Finish ==========')

    # Isolation Forest
    f1_pre = autoencoder1.predict(X_train)

    clf = IsolationForest(random_state = 42, contamination = 0.3)
    clf.fit(f1_pre)
    pred = clf.predict(f1_pre)

    p1 = f1_pre[pred==1]
    p2 = f1_pre[pred==-1]

    # 2st Autoencoder
    autoencoder2 = nn_autoencoder_model(p1)
    autoencoder2.compile(optimizer='adam', 
                         loss='mse',
                         metrics=['accuracy'])

    history2 = autoencoder2.fit(p1, p1,
            epochs=200,
            batch_size=10,
            shuffle=True,
            validation_split=0.2,
            callbacks = [early_stopping,
                          ModelCheckpoint(MODLE_PATH + 'iso_2auto_2nd.h5',monitor='val_loss', save_best_only=True, mode='min')])
    
    print('========== Second Autoencoder Finish ==========')
    # plot the training losses
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history1.history['loss'], 'b', label='1st autoencoder', linewidth=2)
    ax.plot(history2.history['loss'], 'r', label='2nd autoencoder', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(GRAPH_PATH + '2Auto_loss.png')

    # plot the loss distribution of the training set
    X_pred = autoencoder2.predict(autoencoder1.predict(X_train))
    X_pred = pd.DataFrame(X_pred)

    scored = pd.DataFrame()
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
    plt.xlim([0.0,.5])
    plt.savefig(GRAPH_PATH + '2Auto_loss_mae.png')

    return (autoencoder1, autoencoder2)