import os
import numpy as np
import tensorflow as tf
import pandas as pd
import math
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from keras.models import load_model
from train import load_data, create_dataset, seqLen, classification_split, regression_split, create_dataset_multi_feature
import pickle
import copy
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.preprocessing import MinMaxScaler

regression_scaler = MinMaxScaler(feature_range=(0, 1))
classification_scaler = MinMaxScaler(feature_range=(0, 1))

def sarima_test():
    data = pd.read_csv('../data/CGMData.csv', header=None)
    cgm, bolus = load_data(path=os.path.abspath('..'))
    data.reindex(index=data.index[::-1])
    t=0
    for i in range(len(data.columns)):
      data[i][0] = t
      t+=5
    data.iloc[1] = data.iloc[1].interpolate()
    data = data.T
    data = data[:len(bolus)]
    data.columns=["time","CGMreading"]
    y = data.set_index(["time"])
    results = SARIMAXResults.load(os.path.join('..', 'model', "mc_sarima.pkl"))
    classification_model = pickle.load(open(os.path.join('..', 'model', 'classifer_model.pkl'), 'rb'))

    pred = results.get_prediction(start=26255, dynamic=False)
    pred_ci = pred.conf_int()
    ax = y.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Retail_sold')
    plt.legend()
    plt.show()

    y_forecasted = pred.predicted_mean
    y_truth = y[131275:]['CGMreading']
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error is {}'.format(round(mse, 2)))
    print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

    testPredict = []
    for i in range(0,(len(pred.predicted_mean)-6)):
        temp_predict = []
        temp_predict.append(pred.predicted_mean.iloc[i])
        temp_predict.append(pred.predicted_mean.iloc[i+1])
        temp_predict.append(pred.predicted_mean.iloc[i+2])
        temp_predict.append(pred.predicted_mean.iloc[i+3])
        temp_predict.append(pred.predicted_mean.iloc[i+4])
        temp_predict.append(pred.predicted_mean.iloc[i+5])
        testPredict.append(temp_predict)
    testPredict = np.array(testPredict)

    test = cgm[26249:(len(cgm))]

    test = test.reshape(-1, 1)
    test = regression_scaler.fit_transform(test)



    testX, testY = create_dataset_multi_feature(test, 6)

    _, _, _, bolus_test, classification_scaler = classification_split(cgm, bolus)
    bolus_test = bolus_test[6 - 1:testX.shape[0] + 6 - 2]

    predicted_labels = []
    regression_predictions = copy.deepcopy(testPredict)
    classification_train_data = classification_scaler.transform(regression_predictions)
    prediction = classification_model.predict(classification_train_data)
    predicted_labels.extend(prediction)

    print(classification_report(bolus_test, predicted_labels))

if __name__ == '__main__':
    sarima_test()
