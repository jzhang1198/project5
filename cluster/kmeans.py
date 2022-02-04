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

        return


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

        centroids = self._initialize(mat)
        iteration = 0

        while iteration <= 100:

            distances = cdist(centroids,mat,self.metric) #compute a distance matrix where ij represents the distance between row vector i (from centers) and row vector j (from mat)
            assignments = self._assign(distances)
            centroids_new = self._compute_centroids(mat, assignments)

            iteration += 1
            if centroids_new == centroids: #NOTE: WILL HAVE TO ENSURE THAT ORDERING OF DATA IS THE SAME BETWEEN THESE TWO
                return assignments

            else:
                centroids = centroids_new

        return assignments

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
        centroid_indices = np.random.choice(observations, size = self.k, replace = False) #randomly pick three observation indices without replacement
        centroids = mat[center_indices,:]
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

            if len(assignment) == 1:
                assignments[column_index] = assignment

            else: #in the unlikely chance that an observation is equidistant two or more centers, assign randomly
                assignments[column_index] = np.random.choice(assignment))

        return assignments

    def _compute_centroids(self, mat: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        pass



    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
