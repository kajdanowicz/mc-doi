import numpy as np
import math
import copy
import pickle
import model.ccMatrix as ccMatrix
import model.aMatrix as aMatrix
import model.tMatrix as tMatrix


class model():

    def __init__(self):
        self.contagionCorrelationMatrix = ccMatrix.ccMatrix()
        self.adjacencyMatrix = aMatrix.aMatrix()
        self.thresholdsMatrix = tMatrix.tMatrix()
        self.stateMatrix = None  # macierz indykatorow
        self.activityIndexVector = None  # wykladnik

    def fit(self, data, batchType, batchSize):
        # TODO Implement this method
        if self.contagionCorrelationMatrix.matrix is None:
            self.estimateContagionCorrelationMatrix(data)
        if self.adjacencyMatrix.matrix is None:
            self.estimateAdjacencyMatrix(data)
        if batchType == 'time':
            self.thresholdsMatrix.estimateTimeBatch(data)
        elif batchType == 'volume':
            self.thresholdsMatrix.estimateVolumeBatch(data, self.adjacencyMatrix, self.contagionCorrelationMatrix,
                                                      batchSize)
        elif batchType == 'hybrid':
            self.thresholdsMatrix.estimateHybrideBatch(data)
        # TODO stateMatrix and activityIndexVector

    def estimateContagionCorrelationMatrix(self,data):
        self.contagionCorrelationMatrix.estimate(data)

    def estimateAdjacencyMatrix(self,data):
        self.adjacencyMatrix.estimate(data)

    def toPickle(self, directory):
        pickle.dump(self, open(directory + 'model.p', 'wb'))

    @staticmethod
    def fromPickle(directory):
        return pickle.load(open(directory+'model.p','rb'))

    def predict(self):
        # TODO Implement this method
        pass

    def assignContagionsCorrelationMatrix(self, matrix):
        # TODO Implement this method
        if self.stateMatrix is None:
            self.contagionCorrelationMatrix = matrix
        else:
            if self.stateMatrix.shape[1] == matrix.shape[1]:
                self.contagionCorrelationMatrix = matrix

    def assignAdjacencyMatrix(self, adjacencyMatrix):
        # TODO Implement this method
        pass

    def assignThresholdsMatrix(self, thresholdsVector):
        # TODO Implement this method
        pass

    def assignStateMatrix(self, stateMatrix):
        # TODO Implement this method
        pass

    def assignActivityIndexVector(self, activityIndexVector):
        # TODO Implement this method
        pass

    def modelIteration(self):
        # TODO Implement this method
        pass
