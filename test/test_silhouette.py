# write your silhouette score unit tests here
import pytest
from cluster import Silhouette
import numpy as np

def test_privates():
    """
    test private methods called in score public method
    """

    silhouette = Silhouette()
    mat = np.array([[0,0],[0,1],[1,0],[1,1],[100,100],[100,101],[101,100],[101,101]])
    assignments = np.concatenate((np.zeros(int(mat.shape[0] / 2)), np.ones(int(mat.shape[0] / 2))))

    centroids = silhouette._compute_centroids(mat,assignments)
    assert set([tuple(row) for row in centroids]) == {(.5,.5),(100.5,100.5)} #assert that centroids computed by _compute_centroids method are correct

def test_score():
    """
    test score method
    """
    silhouette = Silhouette()
    mat = np.array([[0,0],[1,1],[100,100],[101,101]])
    assignments = np.concatenate((np.zeros(int(mat.shape[0] / 2)), np.ones(int(mat.shape[0] / 2))))

    scores = silhouette.score(mat,assignments)
    tolerance = 10e-6
    score1 = (((np.sqrt(20402) + np.sqrt(20000)) / 2) - np.sqrt(2)) / ((np.sqrt(20402) + np.sqrt(20000)) / 2)
    score2 = (((np.sqrt(19602) + np.sqrt(20000)) / 2) - np.sqrt(2)) / ((np.sqrt(19602) + np.sqrt(20000)) / 2)
    scores_actual = np.array([score1,score2,score2,score1])

    for index in range(0,len(scores)):
        assert abs(scores[index] - scores_actual[index]) < tolerance
