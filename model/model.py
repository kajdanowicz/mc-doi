import numpy as np
import math
import copy


class model():

    def __init__(self):
        self.contagionCorrelationMatrix = None
        self.adjacencyMatrix = None
        self.thresholdsMatrix = None
        self.stateMatrix = None # macierz indykatorow
        self.activityIndexVector = None # wykladnik

    def fit(self,data):
        #TODO Implement this method
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