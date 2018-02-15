import numpy as np
import math


class CorrelationMatrix():
    def __init__(self):
        self.size = 0
        self.correlationMatrix = np.eye(N=self.size)
        print('Init')

    def generateRandomCorrelationMatrix(self,size):
        """Function to generate random correlation matrix.

            Args:
                param1 (int): The first parameter.
                param2 (str): The second parameter.

            Returns:
                bool: The return value. True for success, False otherwise.
            """
        self.size = size
        C = np.random.random((self.size, self.size))
        C = C * 2 - 1
        C *= np.tri(*C.shape, k=-1)
        self.correlationMatrix = C + np.transpose(C) + np.eye(N=self.size)

    def estimateCorrelationMatrixFromData(self,eventsLog):
        self.size = eventsLog['tagID'].max()+1
        self.correlationMatrix = np.eye(N=self.size)
        numUsers=eventsLog['user'].max()+1
        tmp = eventsLog[['user', 'tagID']].drop_duplicates(subset=None, keep='first', inplace=False)
        allUsersIDs = np.linspace(0, numUsers, num=numUsers + 1).astype(int)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                i_users = tmp[tmp['tagID'] == i]['user'].as_matrix()
                j_users = tmp[tmp['tagID'] == j]['user'].as_matrix()
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
                self.correlationMatrix[i][j] = wynik
                self.correlationMatrix[j][i] = wynik

    def testSymetry(self):
        for i in range(self.size):
            for j in range(i+1,self.size):
                if self.correlationMatrix[i][j]!=self.correlationMatrix[j][i]:
                    return False

        return True