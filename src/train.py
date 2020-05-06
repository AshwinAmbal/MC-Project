import numpy as np
import pandas as pd
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle
import sys

np.random.seed(42)

path = os.path.abspath('..')

seqLen = 6
regression_scaler = MinMaxScaler(feature_range=(0, 1))
classification_scaler = MinMaxScaler(feature_range=(0, 1))

def load_data(path=path):
    cgm_df = pd.read_csv(os.path.join(path, 'data', 'CGMData.csv'), header=None)
    cgm_df = cgm_df.T
    cgm_df = cgm_df.iloc[:, 1]
    cgm_df = cgm_df.fillna(0)
    cgm_dataset = cgm_df.values
    cgm_dataset = cgm_dataset.astype('float32')
    cgm_dataset = cgm_dataset[::-1]

    bolus_df = pd.read_csv(os.path.join(path, 'data', 'BolusData.csv'), header=None)
    bolus_df = bolus_df.T
    bolus_df = bolus_df.iloc[:, 1]
    bolus_df = bolus_df.fillna(0)
    bolus_dataset = bolus_df.values
    bolus_dataset = bolus_dataset.astype('float32')
    bolus_dataset = np.array([1 if val > 1 else 0 for val in bolus_dataset])
    bolus_dataset = bolus_dataset[::-1]

    cgm_dataset = cgm_dataset[:bolus_dataset.shape[0]]

    return cgm_dataset, bolus_dataset


def regression_split(cgm_dataset, train_per=0.67):
    train_size = int(len(cgm_dataset) * train_per)
    train, test = cgm_dataset[0:train_size], cgm_dataset[train_size:len(cgm_dataset)]
    train = train.reshape(-1, 1)
    test = test.reshape(-1, 1)
    train = regression_scaler.fit_transform(train)
    test = regression_scaler.transform(test)
    return train, test, regression_scaler


def classification_split(cgm_dataset, bolus_dataset, train_per=0.67):
    train_size = int(len(cgm_dataset) * train_per)
    cgm_train, cgm_test = cgm_dataset[0:train_size], cgm_dataset[train_size:len(cgm_dataset)]
    bolus_train, bolus_test = bolus_dataset[0:train_size], bolus_dataset[train_size:len(bolus_dataset)]

    # Scale Data
    cgm_train = cgm_train.reshape(-1, 1)
    cgm_train = classification_scaler.fit_transform(cgm_train)
    cgm_test = cgm_test.reshape(-1, 1)
    cgm_test = classification_scaler.transform(cgm_test)
    return cgm_train, cgm_test, bolus_train, bolus_test, classification_scaler


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        out = dataset[i + look_back:(i + look_back + look_back), 0]
        if len(out) < look_back:
            break
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def create_dataset_multi_feature(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        out = dataset[i + look_back:(i + look_back + look_back), 0]
        if len(out) < look_back:
            break
        dataX.append(a)
        dataY.append(out)
    return np.array(dataX), np.array(dataY)


def regression_train(multi_feature_pred=False):
    dataset, _ = load_data(path=path)
    train, test, _ = regression_split(dataset)
    if multi_feature_pred:
        trainX, trainY = create_dataset_multi_feature(train, seqLen)
    else:
        trainX, trainY = create_dataset(train, seqLen)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    n_outputs = trainY.shape[1] if multi_feature_pred else 1
    n_timesteps, n_features = trainX.shape[2], trainX.shape[1]
    model = Sequential()
    if multi_feature_pred:
        model.add(LSTM(200, activation='relu', input_shape=(n_features, n_timesteps)))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='relu'))
    else:
        model.add(LSTM(50, activation='relu', input_shape=(n_features, n_timesteps)))
        model.add(Dense(n_outputs, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

    model.save(os.path.join(path, 'model', "mc_lstm.h5"))


def CountLabels(labels):
    count_labels = dict()
    for label in labels:
        if label not in count_labels:
            count_labels[label] = 1
        else:
            count_labels[label] += 1
    return count_labels


def BalanceDataset(data, labels):
    labels = labels.tolist()
    data = data.tolist()
    count_labels = CountLabels(labels)
    count_reqd = sys.maxsize
    for label in count_labels:
        count_reqd = min(count_labels[label], count_reqd)
    count_added = {0: 0, 1: 0}
    balanced_data = []
    balanced_labels = []
    for dat, lab in zip(data, labels):
        if count_added[lab] < count_reqd:
            balanced_data.append(dat)
            balanced_labels.append(lab)
            count_added[lab] += 1
    return np.array(balanced_data), np.array(balanced_labels)


def classification_train():
    cgm, bolus = load_data(path=path)
    cgm_train, _, bolus_train, _, _ = classification_split(cgm, bolus)
    trainX, _ = create_dataset_multi_feature(cgm_train, seqLen)
    trainX = trainX[1:]
    bolus_train = bolus_train[seqLen-1:trainX.shape[0]+seqLen-1]
    trainX, bolus_train = BalanceDataset(trainX, bolus_train)
    clf = LogisticRegression(random_state=42)
    clf.fit(trainX, bolus_train)
    pickle.dump(clf,
                open(os.path.join(path, 'model', 'classifer_model.pkl'), 'wb'))


if __name__ == '__main__':
    regression_train(multi_feature_pred=True)
    classification_train()
