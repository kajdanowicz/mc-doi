import numpy as np
import math


class model():

    def __init__(self):
        self.contagionCorrelationMatrix = None
        self.adjacencyMatrix = None
        self.thresholdsMatrix = None
        self.stateMatrix = None
        self.activityIndexVector = None
        self.numContagions = None

    def estimateParametersFromData(self,data):
        # TODO Implement this method
        self.estimateAdajacencyMatrix(data)

    def estimateContagionCorrelationMatrix(self,data):
        if data.contagionIDDict is None:
            data.addContagionID()
        self.contagionCorrelationMatrix= np.eye(N=data.numContagions)
        numUsers = data.numUsers
        tmp = data.eventLog[['user', 'contagionID']].drop_duplicates(subset=None, keep='first', inplace=False)
        allUsersIDs = np.linspace(0, numUsers, num=numUsers + 1).astype(int)
        for i in range(data.numContagions):
            for j in range(i + 1, data.numContagions):
                i_users = tmp[tmp['contagionID'] == i]['user'].as_matrix()
                j_users = tmp[tmp['contagionID'] == j]['user'].as_matrix()
                ni_users = np.setdiff1d(allUsersIDs, i_users)
                nj_users = np.setdiff1d(allUsersIDs, j_users)
                pij = len(np.intersect1d(i_users, j_users)) / numUsers
                pinj = len(np.intersect1d(i_users, nj_users)) / numUsers
                pnij = len(np.intersect1d(ni_users, j_users)) / numUsers
                pi = len(i_users) / numUsers
                pj = len(j_users) / numUsers
                pni = len(ni_users) / numUsers
                pnj = len(nj_users) / numUsers
                wynik = pij / math.sqrt(pi * pj) - (pnij / math.sqrt(pni * pj) + pinj / math.sqrt(pi * pnj)) / 2
                self.contagionCorrelationMatrix[i][j] = wynik
                self.contagionCorrelationMatrix[j][i] = wynik
        self.numContagions=data.numContagions
        # review

    def verifyContagionCorrelationMatrixSymetry(self):
        for i in range(self.numContagions):
            for j in range(i+1,self.numContagions):
                if self.contagionCorrelationMatrix[i][j]!=self.contagionCorrelationMatrix[j][i]:
                    return False
        return True

    def estimateAdjacencyMatrix(self,data):
        # TODO Implement
        pass

    def estimateThresholdsVector(self,data):
        #TODO Implement
        pass

    def assignContagionsCorrelationMatrix(self, contagionsCorrelationMatrix):
        # TODO Implement this method
        pass

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