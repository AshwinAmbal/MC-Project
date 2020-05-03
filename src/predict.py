import os
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.metrics import mean_squared_error, classification_report
from keras.models import load_model
from src.lstm import load_data, create_dataset, seqLen, classification_split, regression_split
import pickle

np.random.seed(42)

path = os.path.abspath('..')


def get_pred_samples(model, test_sample):
    length_of_test = len(test_sample)
    test_sample = np.array([test_sample])
    for i in range(length_of_test):
        # print("Test Sample {}: ".format(i), test_sample)
        pred_sample = np.reshape(test_sample, (test_sample.shape[0], 1, test_sample.shape[1]))
        testPredict = model.predict(pred_sample)
        # print("Predicted Value {}: ".format(i), testPredict.tolist()[0])
        test_sample = np.array([test_sample[0].tolist()[1:] + testPredict.tolist()[0]])
    return test_sample


def make_predictions():
    cgm, bolus = load_data(path=path)

    _, test, regression_scaler = regression_split(cgm)

    testX, testY = create_dataset(test, seqLen, train=False)
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
    testPredictPlot = np.empty_like(test)
    testPredictPlot[:, :] = np.nan
    # plot baseline and predictions
    plt.plot(regression_scaler.inverse_transform(test))
    plt.plot(testPredict)
    plt.show()

    classification_model = pickle.load(open(os.path.join(path, 'model', 'classifer_model.pkl'), 'rb'))
    _, cgm_test, _, bolus_test, classification_scaler = classification_split(cgm, bolus)
    testX, _ = create_dataset(cgm_test, seqLen, train=True)
    bolus_test = bolus_test[seqLen+1:-1]
    testX = testX[1:]
    bolus_pred = classification_model.predict(testX)
    print(classification_report(bolus_test, bolus_pred))


def predict_regress_classify():
    cgm, bolus = load_data(path=path)
    _, test, regression_scaler = regression_split(cgm)
    testX, testY = create_dataset(test, seqLen, train=False)
    regression_model = load_model(os.path.join(path, 'model', "mc_lstm.h5"))
    classification_model = pickle.load(open(os.path.join(path, 'model', 'classifer_model.pkl'), 'rb'))
    _, _, _, bolus_test, classification_scaler = classification_split(cgm, bolus)
    bolus_test = bolus_test[seqLen + 1:]
    predicted_labels = []
    for row in testX:
        test_sample = get_pred_samples(regression_model, row)
        predicted_sample = regression_scaler.inverse_transform(test_sample)
        predicted_sample = predicted_sample.reshape(-1, 1)
        predicted_sample = classification_scaler.transform(predicted_sample)
        predicted_sample = predicted_sample.reshape(predicted_sample.shape[0])
        prediction = classification_model.predict(np.array([predicted_sample]))
        predicted_labels.extend(prediction)
    print(classification_report(bolus_test, predicted_labels))


if __name__ == '__main__':
    # make_predictions()
    predict_regress_classify()
