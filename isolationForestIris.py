# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:31:18 2018

@author: Ankit_Kumar34
"""

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
df = pd.read_csv("IrisCommaSeperated.csv")

outliers_fraction = 0.01
# Take useful feature and standardize them 
data = df
min_max_scaler = preprocessing.StandardScaler()
np_scaled = min_max_scaler.fit_transform(data)
data = pd.DataFrame(np_scaled)
# train isolation forest 
model =  IsolationForest(contamination = outliers_fraction)
model.fit(data)
# add the data to the main  
df['anomaly'] = pd.Series(model.predict(data))
df['anomaly'] = df['anomaly'].map( {1: 0, -1: 1} )
print(df['anomaly'].value_counts())


a = df.loc[df['anomaly'] == 1]
b = df.loc[df['anomaly'] == 0]
fig, ax = plt.subplots()
ax.plot(df,df,color='blue')
ax.scatter(a,df, color='red')
plt.show()
print(a)         
