__authors__ = ['1492383', '1497551', '1491223']
__group__ = 'DL.15-DJ.17'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist

class KNN:
    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if train_data.ndim > 2:
            self.train_data = np.reshape(train_data, (train_data.shape[0], -1))
        else:
            self.train_data = train_data.copy()

        self.train_data = np.asfarray(self.train_data)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        if test_data.ndim > 2:
            test_data = np.reshape(test_data, (test_data.shape[0], -1))
        test_data = np.asfarray(test_data)

        dist = cdist(test_data, self.train_data, 'euclidean')
        self.neighbors = self.labels[np.argsort(dist, axis=1)[..., :k]]


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        classes = np.empty(self.neighbors.shape[0], dtype=object)
        for row in range(self.neighbors.shape[0]):
            values, index, count = np.unique(self.neighbors[row], return_index=True, return_counts=True)
            index2 = np.argsort(index)
            count = count[index2]
            values = values[index2]
            classes[row] = values[np.argmax(count)]

        return classes


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        classes = self.get_class()

        return classes
