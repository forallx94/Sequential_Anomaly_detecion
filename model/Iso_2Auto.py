
import pandas as pd
from keras.layers import Input, Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import numpy as np

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


def iso_2auto(X_train):
    # 1st Autoencoder
    autoencoder1 = nn_autoencoder_model(X_train)
    autoencoder1.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='loss', mode='min')
    history1 = autoencoder1.fit(X_train, X_train,
            epochs=200,
            batch_size=10,
            shuffle=True,
            callbacks = [early_stopping])
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
    autoencoder2.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    history2 = autoencoder2.fit(p1, p1,
            epochs=200,
            batch_size=10,
            shuffle=True,
            callbacks = [early_stopping])
    print('========== Second Autoencoder Finish ==========')
    # plot the training losses
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history1.history['loss'], 'b', label='1st autoencoder', linewidth=2)
    ax.plot(history2.history['loss'], 'r', label='2nd autoencoder', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig('./result/2Auto_loss.png')

    # plot the loss distribution of the training set
    X_pred = autoencoder2.predict(autoencoder1.predict(X_train))
    X_pred = pd.DataFrame(X_pred)

    scored = pd.DataFrame()
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
    plt.xlim([0.0,.5])
    plt.savefig('./result/2Auto_loss_mae.png')

    return (autoencoder1, autoencoder2)