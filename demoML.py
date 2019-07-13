# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:00:35 2018

@author: Ankit_Kumar34
"""
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
# Isolation Forest
df = pd.read_csv("AuditLogDataA.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
# the hours and if it's night or day (00:00-12:00)

df['DayOfTheWeek'] = df.timestamp.dt.dayofweek
df['Admin'] = (df['DayOfTheWeek'] > 5).astype(int)
df['hours'] = df.timestamp.dt.hour

df['Developer'] = ((df['hours'] >= 00) & (df['hours'] <= 12) & (df['Admin'] == 0) ).astype(int)
df['Tester'] =    ((df['hours'] < 00) &  (df['hours'] > 12) & (df['Admin'] == 0) ).astype(int)
df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
# An estimation of anomly population of the dataset (necessary for several algorithm)
outliers_fraction = 0.01
# Take useful feature and standardize them 
data = df[['value', 'hours', 'Developer', 'Tester', 'Admin']]
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

# visualisation of anomaly throughout time (viz 1)
fig, ax = plt.subplots()

a = df.loc[df['anomaly'] == 1, ['time_epoch', 'value']] #anomaly
print(a)
ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'],a['value'], color='red')
plt.show()