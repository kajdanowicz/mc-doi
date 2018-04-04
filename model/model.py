import numpy as np
import math
import copy
import pickle
import model.ccMatrix as ccMatrix
import model.aMatrix as aMatrix
import model.tMatrix as tMatrix
from model.singleIterResult import singleIterResult
from model.results import results


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
            self.thresholdsMatrix.estimateTimeBatch(data, self.adjacencyMatrix, self.contagionCorrelationMatrix,
                                                      batchSize)
        elif batchType == 'volume':
            self.thresholdsMatrix.estimateVolumeBatch(data, self.adjacencyMatrix, self.contagionCorrelationMatrix,
                                                      batchSize)
        elif batchType == 'hybrid':
            self.thresholdsMatrix.estimateHybrideBatch(data)
        self.fillStateMatrix(data)

    def fillStateMatrix(self,data):
        self.stateMatrix = singleIterResult()
        self.stateMatrix.numContagions = data.numContagions
        self.stateMatrix.numUsers = data.numUsers
        self.stateMatrix.matrix = np.full((self.stateMatrix.numUsers, self.stateMatrix.numContagions), False, dtype=bool)
        for index, row in data.eventLog.iterrows():
            self.stateMatrix.matrix[row['user']][row['contagionID']] = True
        self.activityIndexVector = np.sum(self.stateMatrix.matrix, axis=1)

    def estimateContagionCorrelationMatrix(self,data):
        self.contagionCorrelationMatrix.estimate(data)

    def estimateAdjacencyMatrix(self,data):
        self.adjacencyMatrix.estimate(data)

    def toPickle(self, directory):
        pickle.dump(self, open(directory + 'model.p', 'wb'))

    @staticmethod
    def fromPickle(directory):
        return pickle.load(open(directory+'model.p','rb'))

    def predict(self, numIterations):
        numActivations = 0
        r = results()
        self.adjacencyMatrix.transpose()
        for l in range(numIterations):
            U = self.adjacencyMatrix.matrixTransposed.dot(self.stateMatrix.matrix)
            F = U.dot(self.contagionCorrelationMatrix.matrix) / self.contagionCorrelationMatrix.numContagions
            temp = np.greater_equal(F, self.thresholdsMatrix.matrix)  # porównanie funkcji aktywacji z progiem
            ### dodawanie rekordów bez przekroczenia progu
            for i in np.unique(np.where(temp[:, :] == True)[0]):  # iteracja po użytkownikach, którzy mają przekroczony próg
                temp1 = np.where(temp[i, :] == True)[0]  # tagi, w których dla użytkownika i przekroczony był próg
                temp2 = np.where(self.stateMatrix.matrix[i][:] == True)[0]  # tagi juz aktywne
                temp1 = np.setdiff1d(temp1, temp2)  # usuniecie juz aktywnych tagow
                if (not np.any(self.contagionCorrelationMatrix.matrix[temp1[:, None], temp1] < 0)) and (not temp1.size == 0):  # sprawdzenie, czy kandydaci do aktywacji nie są negatywnie skorelowani
                    # print('YES! ',l)
                    self.stateMatrix.matrix[i][temp1] = True  # aktywacja uzytkownika i w tagach z listy temp1
                    self.activityIndexVector[i] += 1  # Y[i]+=1 #zwiekszenie licznika aktywacji uzytkownika i
                    numActivations += 1
                    for contagion in range(self.stateMatrix.numContagions): #temporary solution
                        self.thresholdsMatrix.matrix[i][contagion] = 1 - math.pow(1 - self.thresholdsMatrix.initialMatrix[i][contagion], self.activityIndexVector[i] + 1)  # aktualizacja thety
            r.addResult(self.stateMatrix)
        print(numActivations)
        return r

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
