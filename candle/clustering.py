import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Clustering:
    def __init__(self):
        pass

class K_means(Clustering):
    def __init__(self):
        super().__init__()
        self.centroids = None

    def gen_initial_centroids(self, X, no_clusters):
        self.centroids = np.random.random((no_clusters, X.shape[1])) #randomly init centroids
        return self.centroids

    def points_to_ownerships(self, X, old_ownerships=None):
        no_centroids = self.centroids.shape[0]
        distances = np.zeros((X.shape[0], no_centroids))

        for i in range(no_centroids):
            diff = X - self.centroids[i,:]
            dist = np.linalg.norm(diff, axis=1)
            distances[:,i] = dist #find euclidean distance to point from all centroids
        ownerships = np.argmin(distances, axis=1) #flatten the arrays into single index per point by summation

        if np.array_equal(ownerships, old_ownerships):
            print("Converged")
            return ownerships, True
        return ownerships, False

    def k_means(self, X, no_clusters, no_iterations):
        self.gen_initial_centroids(X, no_clusters)
        for it in range(no_iterations):
            ownerships, flag = self.points_to_ownerships(X, old_ownerships=None if it==0 else ownerships)
            for i in range(no_clusters):
                points = X[ownerships == i, :]
                if points.shape[0] > 0:
                    self.centroids[i,:] = np.mean(points, axis=0)
            print(f"Iteration {it+1} completed. Centroids {self.centroids}")
            if flag:
                break
        return self.centroids, ownerships
    

    def plot_2d(self, X, centroids, ownerships):
        cols = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
        fig = plt.figure(0)
        plt.grid(True)
        for i in range(X.shape[0]):
            plt.scatter(X[i,0],X[i,1], marker = '.', c = cols[ownerships[i]%len(cols)])
        for i in range(centroids.shape[0]):    
            plt.scatter(centroids[i,0],centroids[i,1],marker = '*', c = cols[i%len(cols)])
        plt.savefig("clusters.png")

X, y = make_blobs(n_samples=1000, n_features=2, centers=4, random_state=23)

clusters = K_means()
centroids, indices = clusters.k_means(X, no_clusters=4, no_iterations=100)
clusters.plot_2d(X, centroids, indices)

