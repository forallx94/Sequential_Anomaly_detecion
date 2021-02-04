from keras.models import  Model
from keras import regularizers
from keras.layers import Dense, Input, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Autoencoder model.
def nn_autoencoder(X_train):
  train_dataset = X_train.copy()
  input_dim = train_dataset.shape[1]
  input = Input(shape=(input_dim, ))
  encode = Dense(input_dim//3*2, activation='relu',kernel_regularizer=regularizers.l2(0.01))(input)
  encode = BatchNormalization()(encode)
  encode = Dense(input_dim//3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encode)
  encode = BatchNormalization()(encode)
  encode = Dense(1, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encode)
  encode = BatchNormalization()(encode)

  decode = Dense(input_dim//3, activation='relu')(encode)
  decode = BatchNormalization()(decode)
  decode = Dense(input_dim//3*2, activation='relu')(decode)
  decode = BatchNormalization()(decode)
  decode = Dense(input_dim, activation='sigmoid')(decode)

  autoencoder = Model(input, decode)

  autoencoder.compile(optimizer='adam',
             loss='mse',
             metrics=['accuracy'])

  early_stopping = EarlyStopping(monitor='loss', mode='min')
  # Train model.
  history = autoencoder.fit(train_dataset, train_dataset,
                          epochs=200,
                          batch_size=128,
                          shuffle=True,
                          callbacks = [early_stopping]
                          )

  # plot the training losses
  fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
  ax.plot(history.history['loss'], 'b', label='Train', linewidth=2)
  ax.set_title('Model loss', fontsize=16)
  ax.set_ylabel('Loss (mae)')
  ax.set_xlabel('Epoch')
  ax.legend(loc='upper right')
  plt.savefig('./result/NN_loss.png')

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
  plt.savefig('./result/NN_loss_distribution.png')

  return autoencoder