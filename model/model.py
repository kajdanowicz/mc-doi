import numpy as np
import math
import copy
import model.ccMatrix as ccMatrix
import model.aMatrix as aMatrix
import model.tMatrix as tMatrix


class model():

    def __init__(self):
        self.contagionCorrelationMatrix = None
        self.adjacencyMatrix = None
        self.thresholdsMatrix = None
        self.stateMatrix = None # macierz indykatorow
        self.activityIndexVector = None # wykladnik

    def fit(self,data):
        #TODO Implement this method
        self.contagionCorrelationMatrix=ccMatrix.ccMatrix()
        self.adjacencyMatrix=aMatrix.aMatrix()
        self.thresholdsMatrix=tMatrix.tMatrix()
        self.contagionCorrelationMatrix.estimate(data)
        self.adjacencyMatrix.estimate(data)
        self.thresholdsMatrix.estimate(data)
        # TODO stateMatrix and activityIndexVector

        pass

    def predict(self):
        #TODO Implement this method
        pass

    def assignContagionsCorrelationMatrix(self, matrix):
        # TODO Implement this method
        if self.stateMatrix is None:
            self.contagionCorrelationMatrix=matrix
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