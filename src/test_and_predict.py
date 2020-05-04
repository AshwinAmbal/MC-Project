import os
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.metrics import mean_squared_error, classification_report
from keras.models import load_model
from src.train import load_data, create_dataset, seqLen, classification_split, regression_split, create_dataset_multi_feature
import pickle
import copy

np.random.seed(42)

path = os.path.abspath('..')


def get_pred_samples(model, test_sample):
    length_of_test = len(test_sample)
    test_sample = np.array([test_sample])
    for i in range(length_of_test):
        pred_sample = np.reshape(test_sample, (test_sample.shape[0], 1, test_sample.shape[1]))
        testPredict = model.predict(pred_sample)
        test_sample = np.array([test_sample[0].tolist()[1:] + testPredict.tolist()[0]])
    return test_sample


def test_regress_classify():
    cgm, bolus = load_data(path=path)

    _, test, regression_scaler = regression_split(cgm)

    testX, testY = create_dataset(test, seqLen)
    regression_model = load_model(os.path.join(path, 'model', "mc_lstm.h5"))

    # make regression predictions
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    testPredict = regression_model.predict(testX)

    # invert predictions
    testPredict = regression_scaler.inverse_transform(testPredict)
    testY = regression_scaler.inverse_transform([testY])

    # calculate root mean squared error
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    test = test.reshape(test.shape[0], 1)
    test = regression_scaler.inverse_transform(test).reshape(test.shape[0])
    # plot baseline and predictions
    plt.plot(test)
    plt.plot(testPredict.reshape(testPredict.shape[0]))
    plt.show()

    classification_model = pickle.load(open(os.path.join(path, 'model', 'classifer_model.pkl'), 'rb'))
    _, cgm_test, _, bolus_test, classification_scaler = classification_split(cgm, bolus)
    testX, _ = create_dataset_multi_feature(cgm_test, seqLen)
    testX = testX[1:]
    bolus_test = bolus_test[seqLen-1:testX.shape[0] + seqLen - 1]
    bolus_pred = classification_model.predict(testX)
    print(classification_report(bolus_test, bolus_pred))


def predict_regress_classify(multi_feature_pred=False):
    cgm, bolus = load_data(path=path)
    _, test, regression_scaler = regression_split(cgm)
    if multi_feature_pred:
        regression_model = load_model(os.path.join(path, 'model', "mc_lstm_multi_feature.h5"))
        testX, testY = create_dataset_multi_feature(test, seqLen)
    else:
        regression_model = load_model(os.path.join(path, 'model', "mc_lstm.h5"))
        testX, testY = create_dataset(test, seqLen)
    classification_model = pickle.load(open(os.path.join(path, 'model', 'classifer_model.pkl'), 'rb'))
    _, _, _, bolus_test, classification_scaler = classification_split(cgm, bolus)
    predicted_labels = []

    if multi_feature_pred:
        reshaped_testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        predicted_regression_values = regression_model.predict(reshaped_testX)
    else:
        predicted_regression_values = []
        for row in testX:
            test_sample = get_pred_samples(regression_model, row)
            predicted_regression_values.extend(test_sample.tolist())
    predicted_regression_values = np.array(predicted_regression_values[:-1])

    testX = testX[1:]
    bolus_test = bolus_test[seqLen - 1:testX.shape[0] + seqLen - 1]

    predicted_regression_values = regression_scaler.inverse_transform(predicted_regression_values)
    testX = regression_scaler.inverse_transform(testX)
    regression_predictions = copy.deepcopy(predicted_regression_values)

    predicted_regression_values = predicted_regression_values.reshape(
                                                 predicted_regression_values.shape[0]*predicted_regression_values.shape[1])
    testX = testX.reshape(testX.shape[0]*testX.shape[1])

    testScore = math.sqrt(mean_squared_error(testX, predicted_regression_values))
    print('Test Score: %.2f RMSE' % (testScore))

    plt.plot(testX)
    plt.plot(predicted_regression_values)
    plt.show()

    classification_train_data = classification_scaler.transform(regression_predictions)
    # classification_train_data = classification_train_data.reshape(classification_train_data.shape[0])
    prediction = classification_model.predict(classification_train_data)
    predicted_labels.extend(prediction)

    print(classification_report(bolus_test, predicted_labels))


if __name__ == '__main__':
    # test_regress_classify()
    predict_regress_classify(multi_feature_pred=False)
