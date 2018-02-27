import numpy as np
import math


class ccMatrix():

    def __init__(self):
        self.matrix=None
        self.numContagions=None

    def estimate(self,data):
        # if data.contagionIDDict is None:
        #     data.addContagionID()
        self.numContagions=data.numContagions
        self.matrix= np.eye(N=self.numContagions)
        numUsers = data.numUsers
        tmp = data.eventLog[['user', 'contagion']].drop_duplicates(subset=None, keep='first', inplace=False)
        allUsersIDs = np.linspace(0, numUsers, num=numUsers + 1).astype(int)
        for i in range(self.numContagions):
            for j in range(i + 1, self.numContagions):
                i_users = tmp[tmp['contagion'] == i]['user'].as_matrix()
                j_users = tmp[tmp['contagion'] == j]['user'].as_matrix()
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
                self.matrix[i][j] = wynik
                self.matrix[j][i] = wynik
        # review

    def verifyMatrixSymetry(self,matrix=None):
        if matrix is None:
            for i in range(self.numContagions):
                for j in range(i+1,self.numContagions):
                    if self.matrix[i][j]!=self.matrix[j][i]:
                        return False
            return True
        else:
            numContagions=matrix.shape[0]
            for i in range(numContagions):
                for j in range(i+1,numContagions):
                    if matrix[i][j]!=matrix[j][i]:
                        return False
            return True

    def assignMatrix(self,matrix):
        #TODO Implement this method
        pass