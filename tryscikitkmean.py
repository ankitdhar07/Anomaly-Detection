import pandas as pd
import numpy as np

import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

df = pd.read_csv('iris.csv')
print(df.head())
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(df)
# Getting the cluster labels
labels = kmeans.predict(df)
# Centroid values
centroids = kmeans.cluster_centers_
# Comparing with scikit-learn centroids
#print(C) # From Scratch
#print(centroids) #
print(df.iloc[0:1,:])