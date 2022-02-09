import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        self.k = k
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        self._training_set = mat
        centroids_old = self._initialize(mat) #initialize clustering algorithm
        error_old = 0
        iteration = 0

        while iteration <= self.max_iter:
            iteration += 1

            distances = cdist(centroids_old,mat,self.metric) #compute a distance matrix where ij represents the distance between row vector i (from centers) and row vector j (from mat)
            assignments = self._assign(distances)
            centroids_new = self._compute_centroids(mat, assignments)
            error_new = self._compute_error(mat, assignments, centroids_new)

            if abs(error_new - error_old) < self.tol: #check if the centroids have stabilized
                self._error = error_new
                self._centroids = centroids_new
                self._assignments = assignments
                return self

            else: #if not, continue
                centroids_old = centroids_new
                error_old = error_new

        self._error = error_old
        self._centroids = centroids_old
        self._assignments = assignments

        return self

    def _initialize(self, mat: np.ndarray) -> np.ndarray:
        """
        yields a np.ndarray containing k randomly selected observations assigned to be initial centroids.

            inputs:
                mat: np.ndarray
                    A 2D matrix where the rows are observations and columns are features

            outputs:
                centroids: np.ndarray
                    A 2D matrix where the rows are centroids and the columns are features
        """

        observation_indices = np.linspace(0,mat.shape[0]-1,mat.shape[0]) #create a np.ndarray to hold indices of observations
        centroid_indices = np.random.choice(observation_indices, size = self.k, replace = False) #randomly pick three observation indices without replacement
        centroids = mat[[int(index) for index in centroid_indices]]
        return centroids

    def _assign(self, distances: np.ndarray) -> np.ndarray:
        """
        assigns observations to clusters based on distance.

        inputs:
            distances: np.ndarray
                A 2D matrix where ij represents the distance between centroid i and observation j

        outputs:
            assignments: np.ndarray
                A 1D array where element i represents the cluster assignment of the ith observation
        """

        assignments = np.zeros(distances.shape[1]) #create an np.ndarray to hold assignments for observations
        for column_index in range(0,distances.shape[1]): #determine which cluster an assignment belongs to based on distance to center
            assignment = np.argmin(distances[:,column_index])
            assignments[column_index] = assignment
        return assignments

    def _compute_centroids(self, mat: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """
        computes centroids of clusters.

        inputs:
            mat: np.ndarray
                A 2D matrix with rows representing observations

            assignments: np.ndarray
                A 1D array where element i represents the cluster assignment of the ith observation
        """
        centroids = np.zeros((self.k, mat.shape[1])) #generate an empty matrix to hold centroids

        for k in range(0,self.k):
            cluster = mat[np.where(assignments == k)] #find observations of the same cluster
            centroid = np.sum(cluster,axis=0) / cluster.shape[0] #compute cluster centroid
            centroids[k] = centroid #add to centroids matrix
        return centroids

    def _compute_error(self, mat: np.ndarray, assignments: np.ndarray, centroids: np.ndarray) -> float:
        """
        computes the mean-squared error of the centroid distances.

        inputs:
            mat: np.ndarray
                A 2D matrix with rows representing observations

            assignments: np.ndarray
                A 1D array where element i represents the cluster assignment of the ith observation

            centroids: np.ndarray
                A 2D matrix where the ith row represents the centroid for cluster i.
        """

        error = 0
        for observation_index in range(0,mat.shape[0]):
            observation = mat[observation_index]
            centroid = centroids[int(assignments[observation_index])] #find the centroid in the same cluster as the observation
            error += np.linalg.norm(observation - centroid) ** 2 #update error

        mse = error / mat.shape[0] #compute mse

        return mse

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if set([tuple(row) for row in mat]) == set([tuple(row) for row in self._training_set]): #check if the observations are the same as the original training set
            return self._assignments

        else: #if observations are not the same as the training set, compute assignments based on the centroids found in fit method
            centroids = self._centroids
            distances = cdist(centroids, mat, self.metric)
            assignments = self._assign(distances)
            return assignments

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self._error

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self._centroids
