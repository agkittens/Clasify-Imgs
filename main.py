import image as classify
import constants
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

cl = classify.Clasifier()


def find_clusters():
    temp = cl.img_feat
    temp, _ = make_blobs(n_samples=300, centers=8, random_state=constants.RANDOM1)

    num = []
    possible_clusters = []
    clusters = range(3, 10)

    for cluster in clusters:
        kmeans = KMeans(n_clusters=cluster, random_state=constants.RANDOM1)
        kmeans.fit(temp)
        num.append(kmeans.inertia_)

    for cluster in clusters:
        kmeans = KMeans(n_clusters=cluster, random_state=constants.RANDOM1)
        kmeans.fit(temp)
        if kmeans.inertia_ < np.mean(num, axis=0):
            possible_clusters.append(cluster)

    return possible_clusters[-2]

cl.clusters = find_clusters()
cl.make_categories()

cl.find_different_imgs()


