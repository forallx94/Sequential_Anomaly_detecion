import numpy as np
from keras.models import  Model, Sequential
from keras.layers import Dense, LSTM , RepeatVector, TimeDistributed, BatchNormalization
import matplotlib.pyplot as plt

def LSTM_Autoencoder(train_dataset):
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

  # fit
  history = model.fit(x, x,
                      epochs=100, batch_size=128)
  
  # plot the training losses
  fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
  ax.plot(history['loss'], 'b', label='Train', linewidth=2)
  ax.set_title('Model loss', fontsize=16)
  ax.set_ylabel('Loss (mae)')
  ax.set_xlabel('Epoch')
  ax.legend(loc='upper right')
  plt.savefig('./result/LSTM_loss.png')

  return model
  
