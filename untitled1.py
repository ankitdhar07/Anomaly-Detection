import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance

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

# calculate with different number of centroids to see the loss plot (elbow method)
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]
fig, ax = plt.subplots()
ax.plot(n_cluster, scores)
plt.show()


# get the distance between each point and its nearest centroid. The biggest distances are considered as anomaly
distance = getDistanceByPoint(data, kmeans[2])
number_of_outliers = int(outliers_fraction*len(distance))
threshold = distance.nlargest(number_of_outliers).min()
# anomaly contain the anomaly result of method 2.1 Cluster (0:normal, 1:anomaly) 
df['anomaly'] = (distance >= threshold).astype(int)
fig, ax = plt.subplots()

a = df.loc[df['anomaly'] == 1, ['time_epoch', 'value']] #anomaly
print(df['anomaly'].value_counts())
print(a)

ax.plot(df['time_epoch'], df['value'], color='blue')
ax.scatter(a['time_epoch'],a['value'], color='red')
plt.show()
