import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(42)

path = os.path.abspath('../data')

dataframe = pd.read_csv(os.path.join(path, 'CGMData.csv'), header=None)
dataframe = dataframe.T
dataframe = dataframe.iloc[:, 1]
dataframe = dataframe.fillna(0)
dataset = dataframe.values
dataset = dataset.astype('float32')

dataset = dataset.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# print(len(train), len(test))

seqLen = 1
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, train=False):
    dataX, dataY = [], []
    # if train:
    #     pad_prev = np.array([[0] for i in range(look_back)])
    #     dataset = np.concatenate((pad_prev, dataset), axis=0)
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# X, y = create_dataset(dataset, look_back=seqLen)
trainX, trainY = create_dataset(train, seqLen, train=True)
testX, testY = create_dataset(test, seqLen, train=False)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, seqLen)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
