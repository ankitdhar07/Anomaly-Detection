import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


# An estimation of anomly population of the dataset (necessary for several algorithm)
#gaussian distribution
#isolation forest 
# one class SVM 
outliers_fraction = 0.1
dictionary={}
def load_dataset(name):
    return np.loadtxt(name)

def euclidian(a, b):
    return np.linalg.norm(a-b)

def plot(dataset, history_centroids, belongs_to):
    colors = ['r', 'g','c','y']
    
    fig, ax = plt.subplots()
    
    for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

    history_points = []
    for index, centroids in enumerate(history_centroids):
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))

                plt.pause(0.8)


def kmeans(k, epsilon=0, distance='euclidian'):
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian
    dataset = load_dataset('Iris (2).csv')
    # dataset = dataset[:, 0:dataset.shape[1] - 1]
    num_instances, num_features = dataset.shape
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    belongs_to = np.zeros((num_instances, 1))
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)

            belongs_to[index_instance, 0] = np.argmin(dist_vec)

        tmp_prototypes = np.zeros((k, num_features))

        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(dataset[instances_close], axis=0)
            # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
            tmp_prototypes[index, :] = prototype

        prototypes = tmp_prototypes

        history_centroids.append(tmp_prototypes)

    plot(dataset, history_centroids, belongs_to)

    return prototypes, history_centroids, belongs_to

def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0, len(data)):
        Xa = data.iloc[i][:]
        Xb = model
        distance.set_value(i, np.linalg.norm(Xa-Xb))
    return distance

def execute():
    dataset = load_dataset('Iris (2).csv')
    centroids, history_centroids, belongs_to = kmeans(3)
   
    k=3
    df = pd.read_csv("Iris (2).csv",delimiter=' ',header=None)
    df[df.shape[1]]=belongs_to.reshape(df.shape[0],1)
    for i in range(k):
        print(i)
        df0=df.copy()
        df0=df0.loc[df0[4]==i]
        del df0[4]
   
        distance=getDistanceByPoint(df0,centroids[i])
        #print(distance)
        number_of_outliers = int(outliers_fraction*len(distance))
        threshold = distance.nlargest(number_of_outliers).min()
        df0['anomaly'] = (distance >= threshold).astype(int)

execute()  




