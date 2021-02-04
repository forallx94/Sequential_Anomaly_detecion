import numpy as np
from keras.models import  Model, Sequential
from keras.layers import Dense, LSTM , RepeatVector, TimeDistributed, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def lstm_autoencoder(train_dataset):
  x = np.expand_dims(train_dataset,axis=1)

  # LSTM Autoencoder
  model = Sequential()

  # Encoder
  model.add(LSTM(x.shape[-1]//3*2, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(x.shape[-1]//3, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(1, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  # model.add(RepeatVector(10))

  # Decoder
  model.add(LSTM(x.shape[-1]//3, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(x.shape[-1]//3*2, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(TimeDistributed(Dense(x.shape[-1])))

  model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

  early_stopping = EarlyStopping(monitor='loss', mode='min')

  # fit
  history = model.fit(x, x,
                      epochs=200, 
                      batch_size=128,
                      callbacks = [early_stopping])
  
  # plot the training losses
  fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
  ax.plot(history.history['loss'], 'b', label='Train', linewidth=2)
  ax.set_title('Model loss', fontsize=16)
  ax.set_ylabel('Loss (mae)')
  ax.set_xlabel('Epoch')
  ax.legend(loc='upper right')
  plt.savefig('./result/LSTM_loss.png')

  # plot the loss distribution of the training set
  X_pred = model.predict(x)
  X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
  X_pred = pd.DataFrame(X_pred)

  scored = pd.DataFrame()
  Xtrain = train_dataset
  scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
  plt.figure(figsize=(16,9), dpi=80)
  plt.title('Loss Distribution', fontsize=16)
  sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
  plt.xlim([0.0,.5])
  plt.savefig('./result/LSTM_loss_distribution.png')

  return model
  
