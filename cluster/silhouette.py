import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """

        self.metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        scores = np.zeros(X.shape[0]) #generate an array to hold silhouette scores
        centroids = self._compute_centroids(X,y)
        observation_indices = np.linspace(0,len(X)-1,len(X))
        ks = list(set(y)) #find unique k values

        for i in observation_indices:
            observation = X[int(i)]

            k = y[int(i)] #find k for the observation
            distances = cdist([observation],X, self.metric)[0] #determine the cluster with centroid closest to observation
            neighbor_k = y[np.argmin([distances[int(i)] if y[int(i)] != k else np.inf for i in observation_indices])]
            
            cluster = X[[int(index) for index in observation_indices if y[int(index)] == k and index != i]] #get observations in the same cluster as observation
            neighbor_cluster = X[[int(index) for index in observation_indices if y[int(index)] == neighbor_k]] #get observations in the cluster closest to observation

            av_distance = sum(cdist([observation],cluster,self.metric)[0]) / cluster.shape[0] #compute silhouette score
            av_neighbor_distance = sum(cdist([observation],neighbor_cluster,self.metric)[0]) / neighbor_cluster.shape[0]
            scores[int(i)] = (av_neighbor_distance - av_distance) / max(av_distance,av_neighbor_distance)

        return scores


    def _compute_centroids(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ks = len(list(set(y)))
        centroids = np.zeros((ks, X.shape[1]))

        for k in range(0,ks):
            cluster = X[np.where(y == k)]
            centroid = np.sum(cluster,axis=0) / cluster.shape[0]
            centroids[k] = centroid

        return centroids
