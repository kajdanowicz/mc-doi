import numpy as np
import math
import pandas as pd
from data.data import data
from tqdm import trange


class cc_matrix():

    def __init__(self):
        self.matrix=None
        self.num_contagions=None
        self.num_users_performing_events = None

    def estimate(self,data):
        """
        2000 contagions ~ 30min
        300 contagions ~ 30s
        @:type data: data
        """
        data.add_contagion_id()
        self.num_contagions=data.num_contagions
        self.matrix= np.eye(N=self.num_contagions)
        self.num_users_performing_events=len(data.event_log.user.unique())
        tmp = data.event_log[['user', 'contagion_id']].drop_duplicates(subset=None, keep='first', inplace=False)
        tmp = pd.merge(tmp[['user', 'contagion_id']], tmp[['user', 'contagion_id']], on='user',suffixes=('_1','_2')).groupby(['contagion_id_1','contagion_id_2']).count()
        for i in range(self.num_contagions):
            count_i = float(tmp.loc[(i, i)].values[0])
            for j in range(i + 1, self.num_contagions):
                count_j = float(tmp.loc[(j, j)].values[0])
                if (i,j) in tmp.index:
                    count_ij = float(tmp.loc[(i, j)].values[0])
                else:
                    count_ij = 0.
                wynik = count_ij / math.sqrt(count_i * count_j) - ((count_j-count_ij) / math.sqrt((self.num_users_performing_events - count_i) * count_j) + (count_i - count_ij) / math.sqrt(count_i * (self.num_users_performing_events - count_j))) / 2
                self.matrix[i][j] = wynik
                self.matrix[j][i] = wynik
        # review

    def verify_matrix_symetry(self, matrix=None):
        if matrix is None:
            for i in range(self.num_contagions):
                for j in range(i+1, self.num_contagions):
                    if self.matrix[i][j]!=self.matrix[j][i]:
                        return False
            return True
        else:
            num_contagions=matrix.shape[0]
            for i in range(num_contagions):
                for j in range(i+1,num_contagions):
                    if matrix[i][j]!=matrix[j][i]:
                        return False
            return True

    def assign_matrix(self, matrix):
        #TODO Implement this method
        pass

    def random_matrix(self, size):
        self.num_contagions = size
        C = np.random.random((self.num_contagions, self.num_contagions))
        C = C * 2 - 1
        C *= np.tri(*C.shape, k=-1)
        self.matrix = C + np.transpose(C) + np.eye(N=self.size)
        # review