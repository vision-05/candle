import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def gen_initial_centroids(X, no_clusters):
    centroids = np.random.random((no_clusters, X.shape[1])) #randomly init centroids
    return centroids

def points_to_ownerships(X, centroids):
    no_centroids = centroids.shape[0]
    distances = np.zeros((X.shape[0], no_centroids))
    for i in range(no_centroids):
        diff = X - centroids[i,:]
        dist = np.linalg.norm(diff, axis=1)
        distances[:,i] = dist #find euclidean distance to point from all centroids
    new = distances - np.min(distances, axis=1, keepdims=True) #set minimum distance to 0, all others positive
    indices = np.arange(new.shape[1]) #create array of indices [0, 1, 2, ..., no_centroids-1]
    new = np.where(new == 0, indices, 0) #set the minimum distance to its index, others to 0
    ownerships = np.sum(new, axis=1) #flatten the arrays into single index per point by summation
    return ownerships

def k_means(X, no_clusters, no_iterations):
    centroids = gen_initial_centroids(X, no_clusters)
    for it in range(no_iterations):
        ownerships = points_to_ownerships(X, centroids)
        for i in range(no_clusters):
            points = X[ownerships == i, :]
            if points.shape[0] > 0:
                centroids[i,:] = np.mean(points, axis=0)
        print(f"Iteration {it+1} completed. Centroids {centroids}")
    return centroids, ownerships
    

def plot_2d(X, centroids, ownerships):
    cols = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    fig = plt.figure(0)
    plt.grid(True)
    for i in range(X.shape[0]):
        plt.scatter(X[i,0],X[i,1], marker = '.', c = cols[ownerships[i]%len(cols)])
    for i in range(centroids.shape[0]):    
        plt.scatter(centroids[i,0],centroids[i,1],marker = '*', c = cols[i%len(cols)])
    plt.savefig("clusters.png")

X, y = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=23)

centroids, indices = k_means(X, no_clusters=4, no_iterations=10)
plot_2d(X, centroids, indices)

