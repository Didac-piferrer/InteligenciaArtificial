__authors__ = ['1492383', '1497551', '1491223']
__group__ = 'DL.15-DJ.17'

import numpy as np
import time
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        self.labels = np.zeros((self.X.shape[0],))
        self._init_centroids()

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """
        if X.ndim > 2:
            self.X = np.reshape(X, (-1, X.shape[-1]))
        else:
            self.X = X.copy()

        self.X = np.asfarray(self.X)

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first':
            uniquePoints, indexUnique = np.unique(self.X, axis=0, return_index=True)  # Devuelve valores unicos ordenador por valor y indices
            indexUnique = np.sort(indexUnique)
            indexUnique = np.resize(indexUnique, self.K)  # Ordenamos indicies y reducimos
            self.centroids = self.X[indexUnique]
            self.old_centroids = np.zeros(self.centroids.shape, dtype="float64")
        if self.options['km_init'].lower() == 'random':
            uniquePoints, indexUnique = np.unique(self.X, axis=0, return_index=True)
            np.random.shuffle(indexUnique)
            indexUnique = np.resize(indexUnique, self.K)
            self.centroids = self.X[indexUnique]
            self.old_centroids = np.zeros(self.centroids.shape, dtype="float64")

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids.copy()
        self.centroids[:] = 0
        for centroidId in range(self.K):
            pointsCentroid = np.where(self.labels == centroidId)[0] # Todos los puntos de X centroide
            self.centroids[centroidId] = np.mean(self.X[pointsCentroid], axis=0) # Media de todos los puntos anteriores

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.old_centroids, self.centroids, rtol=self.options['tolerance'], atol=0, equal_nan=False)  # options['tolerance']

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        iterations = 0
        while True: # Simulación do-while
            self.get_labels()
            self.get_centroids()
            iterations += 1
            if self.converges() or iterations > self.options['max_iter']:
                break
        pass

    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """
        sumWCD = 0
        for centroidId in range(self.K):
            punts = np.argwhere(self.labels == centroidId) # Indice puntos con X centroid
            sumWCD += np.sum((self.X[punts[:, 0]] - self.centroids[centroidId]) ** 2) # Sumatorio distancia puntos con centroid
        sumWCD = sumWCD / self.X.shape[0]
        return sumWCD

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        check = False
        index = 2
        self.__init__(self.X, index)
        self.fit()
        WCDkActual = self.whitinClassDistance()
        while not check and index < max_K:
            index += 1
            self.__init__(self.X, index)
            self.fit()
            WCDkAdelantado = self.whitinClassDistance()
            temp = 100 - 100 * (WCDkAdelantado / WCDkActual)
            WCDkActual = WCDkAdelantado
            if temp < 20:
                check = True
                self.K = index - 1

        if not check:
            self.K = max_K
        return check


def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    coordX = (X[:, 0, np.newaxis] - C[:, 0]) ** 2 # Calculo separado de las 3 componentes para la distancia
    coordY = (X[:, 1, np.newaxis] - C[:, 1]) ** 2
    coordZ = (X[:, 2, np.newaxis] - C[:, 2]) ** 2
    dist = np.sqrt(coordX + coordY + coordZ)
    return dist

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """
    probMatrix = utils.get_color_prob(centroids)
    centroidColors = np.empty(centroids.shape[0], dtype=utils.colors.dtype)

    for indexC in range(centroids.shape[0]):
        indexMax = np.argmax(probMatrix[indexC])
        centroidColors[indexC] = utils.colors[indexMax]

    return centroidColors

