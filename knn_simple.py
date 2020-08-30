"""
https://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html
"""
print(__doc__)

import numpy as np

from sklearn import cluster

n_clusters = 2

k_means = cluster.KMeans(n_clusters=n_clusters)

data = np.array([[1, 2, 3, 7, 8, 9, 20, 21, 22]])  # shape: (1,9)
print(data)
data = data.reshape((9, 1))
k_means.fit(data)
print(k_means.labels_)

k_means = cluster.KMeans(n_clusters=4)
k_means.fit(data)
print(k_means.labels_)


def centers(model, data) -> dict:
    x = clusters(model, data)
    centers = dict()
    for k in x:
        centers[k] = np.mean(x[k])
    return centers


def clusters(model, data):
    x = dict()
    for label in np.unique(model.labels_):
        x[label] = list()
    for i in range(len(data)):
        x[model.labels_[i]].append(data[i][0])
    return x


def calculate_variance(model, data):
    x  = clusters(model, data)
    c = centers(model, data)
    v = 0
    for k in x:
        v = v + np.var(np.array(x[k]) - c[k])
    return v

def create_model(data, n_clusters):
    k_means = cluster.KMeans(n_clusters=n_clusters)
    k_means.fit(data)
    return k_means

variances = np.zeros(8)
for i in range(5):
    model = create_model(data, int(i+1))
    variances[i] = calculate_variance(model, data)

print(variances)