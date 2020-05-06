import csv
import os
import itertools
import pandas as pd
# !pip install statsmodels
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
from pylab import rcParams

def train_sarima():
    data = pd.read_csv('../data/CGMData.csv', header=None)
    bolus = pd.read_csv('../data/BolusData.csv', header=None)
    data.reindex(index=data.index[::-1])
    bolus.reindex(index=bolus.index[::-1])
    bolus = bolus.T
    t =0
    for i in range(len(data.columns)):
      data[i][0] = t
      t+=5
    data.iloc[1] = data.iloc[1].interpolate()
    data = data.T
    data = data[:len(bolus)]
    data.columns=["time","CGMreading"]

    y = data.set_index(["time"])

    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',freq=12)
    fig = decomposition.plot()
    plt.show()

    try:
        mod = sm.tsa.statespace.SARIMAX(y,order=[1,1,1],seasonal_order=[0,0,1,12],enforce_stationarity=False,enforce_invertibility=False)
        results = mod.fit()
        print(results.aic)
    except:
        pass

    results.save(os.path.join('..', 'model', "mc_sarima.pkl"))

if __name__ == '__main__':
    train_sarima()
