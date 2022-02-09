# Write your k-means unit tests here
import pytest
from cluster import KMeans
import numpy as np

def test_privates():
    """
    unit tests for private methods called by the fit and predict public methods
    """
    kmeans = KMeans(2)

    mat = np.array([[0,0],[0,1],[1,0],[1,1],[100,100],[100,101],[101,100],[101,101]])

    initial_centroids = kmeans._initialize(mat) #assert that initialized centroids are in mat
    for centroid in initial_centroids:
        assert centroid in mat

    assignments = kmeans._assign(cdist(initial_centroids,mat,kmeans.metric)) #assert that assignments are in the correct domain
    for assignment in assignments:
        assert assignment in range(0,kmeans.k)

    assignments = np.concatenate((np.zeros(int(mat.shape[0] / 2)), np.ones(int(mat.shape[0] / 2)))) #assert that centroid calculation is done correclty
    centroids = kmeans._compute_centroids(mat,assignments)
    assert set([tuple(row) for row in centroids]) == {(.5,.5),(100.5,100.5)}

    mse = kmeans._compute_error(mat,assignments,centroids) #check that mse calculation is done correctly
    tolerance = 10e-6
    assert abs(mse-.5) < tolerance
