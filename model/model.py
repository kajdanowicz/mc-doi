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


    def estimateAdjacencyMatrix(self,data):
        # TODO Implement
        pass

    def estimateThresholdsVector(self,data):
        #TODO Implement
        indykatory_est = []
        I = np.full((data.numUsers, data.numContagions), False, dtype=bool)
        for i in range(history):
            for index, row in event_log[event_log['ts'] == i].iterrows():
                I[row['userNEW'], row['tagID']] = True
            indykatory_est.append(I)
            I = copy.deepcopy(I)
        pass

    def assignContagionsCorrelationMatrix(self, contagionsCorrelationMatrix):
        # TODO Implement this method
        if self.stateMatrix is None:
            self.contagionCorrelationMatrix=contagionsCorrelationMatrix
        else:
            if self.stateMatrix.shape[1] == contagionsCorrelationMatrix.shape[1]:
                self.contagionCorrelationMatrix = contagionsCorrelationMatrix
        # review

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