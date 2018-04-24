import numpy as np
import math
import pandas as pd
from data.data import Data
from tqdm import trange


class ccMatrix():

    def __init__(self):
        self.matrix=None
        self.num_contagions=None
        self.numUsersPerformingEvents = None

    def estimate(self,data):
        """
        2000 contagions ~ 30min
        300 contagions ~ 30s
        @:type data: data
        """
        data.add_contagion_id()
        self.num_contagions=data.numContagions
        self.matrix= np.eye(N=self.num_contagions)
        self.numUsersPerformingEvents=len(data.eventLog.user.unique())
        tmp = data.eventLog[['user', 'contagionID']].drop_duplicates(subset=None, keep='first', inplace=False)
        tmp = pd.merge(tmp[['user', 'contagionID']], tmp[['user', 'contagionID']], on='user',suffixes=('_1','_2')).groupby(['contagionID_1','contagionID_2']).count()
        for i in range(self.num_contagions):
            count_i = float(tmp.loc[(i, i)].values[0])
            for j in range(i + 1, self.num_contagions):
                count_j = float(tmp.loc[(j, j)].values[0])
                if (i,j) in tmp.index:
                    count_ij = float(tmp.loc[(i, j)].values[0])
                else:
                    count_ij = 0.
                wynik = count_ij / math.sqrt(count_i * count_j) - ((count_j-count_ij) / math.sqrt((self.numUsersPerformingEvents-count_i) * count_j) + (count_i-count_ij) / math.sqrt(count_i * (self.numUsersPerformingEvents-count_j))) / 2
                self.matrix[i][j] = wynik
                self.matrix[j][i] = wynik
        # review

    def verifyMatrixSymetry(self,matrix=None):
        if matrix is None:
            for i in range(self.num_contagions):
                for j in range(i+1, self.num_contagions):
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

    def randomMatrix(self,size):
        self.num_contagions = size
        C = np.random.random((self.num_contagions, self.num_contagions))
        C = C * 2 - 1
        C *= np.tri(*C.shape, k=-1)
        self.matrix = C + np.transpose(C) + np.eye(N=self.size)
        # review