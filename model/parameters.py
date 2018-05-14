from abc import abstractmethod
from data.data import Data
from numpy import ndarray
from collections import defaultdict
import networkx as nx
import numpy as np
import math
import pandas as pd
import copy


class BaseParameter:
    """
    Base class for model's parameters
    """
    def __init__(self):
        self.matrix = None

    @abstractmethod
    def estimate(self, data: Data, **kwargs):
        pass

    @abstractmethod
    def assign_matrix(self,matrix: ndarray):
        pass


class Adjacency(BaseParameter):
    # TODO Implement assign_matrix method
    def __init__(self):
        super(Adjacency, self).__init__()

    def estimate(self, data: Data, **kwargs):
        if data.sorted is False:
            raise NameError('Data not sorted. Can not estimate adjacency matrix.')
        self.num_users_ = data.num_users
        self.v_2_u_ = defaultdict(lambda: 0)
        self.v_and_u_ = defaultdict(lambda: 0)
        self.u_ = defaultdict(lambda: 0)
        self.event_queue_ = dict()
        prev_contagion = None
        # TODO implement whithout networkx library
        data.add_graph()
        d = data.event_log.drop_duplicates(subset=['contagion', 'user'], keep='first')
        for row in d.itertuples(index=False, name=None):
            prev_contagion = self.__verify_contagion(row, prev_contagion)
            self.__propagate(row[0], row[1], data.graph)
        self.__reset_event_queue()
        self._calculate_weights(data.graph)
        self.__clean_counters()

    def assign_matrix(self, matrix):
        #TODO Implement this method
        pass

    def _transpose(self):
        if self.__dict__.get('matrix_transposed_',None) is None:
            self.matrix_transposed_ = self.matrix.transpose()

    def __clean_counters(self):
        # TODO What if dicts do not exist?
        del self.v_2_u_
        del self.v_and_u_
        del self.u_

    def __verify_contagion(self, row: tuple, prev_contagion: int) -> int:
        self.u_[row[1]] += 1
        if row[2] is not prev_contagion:
            self.__reset_event_queue()
            prev_contagion = row[2]
        return prev_contagion

    def _calculate_weights(self, graph):
        G = graph.to_directed()
        nx.set_edge_attributes(G, 0, 'weight')
        for v in self.u_.keys():
            for u in G.successors(v):
                G[v][u]['weight'] = self.__MLE_Jaccard_index(v, u)
        for i in G.nodes():
            in_degree = G.in_degree(i, weight='weight')
            if in_degree != 0:
                for v, u in G.in_edges(i):
                    G[v][u]['weight'] /= in_degree
        self.matrix = nx.adjacency_matrix(G, weight='weight')

    def __MLE_Bernoulli_trial(self, v, u):
        return round(float(self.v_2_u_[(v, u)]) / float(self.u_[v]), 6)

    def __MLE_Jaccard_index(self, v, u):
        if self.v_2_u_[(v, u)] != 0:
            return round(float(self.v_2_u_[(v, u)]) / float(self.u_[v]+self.u_[u]-(self.v_and_u_[(u,v)]+self.v_and_u_[(v,u)])), 6)
        else:
            return 0

    def __reset_event_queue(self):
        self.event_queue_ = dict()

    def add_to_queue(self, user, ts):
        self.event_queue_[user] = ts

    def __propagate(self, ts, user, graph):
        for v in graph.neighbors(user):
            # if self.event_queue_.get(v):
            if v in self.event_queue_:
                self.v_and_u_[(user, v)] += 1
                # self.v_and_u_[(v, user)] += 1
                if ts - self.event_queue_[v] != 0:
                    self.v_2_u_[(v, user)] += 1
        self.add_to_queue(user, ts)


class ContagionCorrelation(BaseParameter):

    def __init__(self):
        super(ContagionCorrelation, self).__init__()
        self.num_contagions=None
        self.num_users_performing_events = None

    def estimate(self,data, **kwargs):
        """
        """
        # TODO Reconsider sparse matrix implementation
        # TODO Consider using PyTables
        # TODO Refactor
        data.add_contagion_id()
        self.num_contagions=data.num_contagions
        self.matrix = np.eye(N=self.num_contagions)
        self.num_users_performing_events=len(data.event_log.user.unique())
        tmp = data.event_log[[Data.user, Data.contagion_id]].drop_duplicates(subset=None, keep='first', inplace=False)
        tmp = pd.merge(tmp[[Data.user, Data.contagion_id]], tmp[[Data.user, Data.contagion_id]], on=Data.user,suffixes=('_1','_2')).groupby([Data.contagion_id+'_1',Data.contagion_id+'_2']).count()
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

    def assign_matrix(self, matrix):
        #TODO Implement this method
        pass

    def verify_matrix_symmetry(self, matrix=None):
        if matrix is None:
            for i in range(self.num_contagions):
                for j in range(i+1, self.num_contagions):
                    if self.matrix[i][j] != self.matrix[j][i]:
                        return False
            return True
        else:
            num_contagions=matrix.shape[0]
            for i in range(num_contagions):
                for j in range(i+1,num_contagions):
                    if matrix[i][j] != matrix[j][i]:
                        return False
            return True


class Threshold(BaseParameter):

    def __init__(self):
        super(Threshold, self).__init__()
        self.initial_matrix = None
        self.num_users = None

    def estimate(self, data: Data, **kwargs):
        # TODO estimate this method - handle different batch types
        batch_type = kwargs.get('batch_type', None)
        pass

    def estimate_volume_batch(self, data, a_matrix, cc_matrix, volume):
        data.add_contagion_id()
        data.construct_event_log_grouped()
        indicators = []
        I = np.full((data.num_users, data.num_contagions), False, dtype=bool)
        event_id = 0
        while event_id < data.event_log[Data.event_id].max():
            for index, row in data.event_log[(data.event_log[Data.event_id] > event_id) & (data.event_log[Data.event_id] <= event_id + volume)].iterrows():
                I[row[Data.user]][row[Data.contagion_id]] = True
            indicators.append(I)
            I = copy.deepcopy(I)
            event_id += volume
        Y = np.sum(indicators[0], axis=1)
        self._estimate(Y, a_matrix, cc_matrix, data, indicators)

    def assign_matrix(self, matrix):
        #TODO Implement this method
        pass

    def _estimate(self, Y, a_matrix, cc_matrix, data, indicators):
        a_matrix._transpose()
        # print('Adjacency.matrix_transposed_.shape', Adjacency.matrix_transposed_.shape)
        # print('indicators[0].shape', indicators[0].shape)
        max_neg = defaultdict(lambda : -2)
        min_pos = defaultdict(lambda : 2)
        for l in range(len(indicators) - 1):
            U = a_matrix.matrix_transposed_.dot(indicators[l])
            F = U.dot(cc_matrix.matrix) / data.num_contagions
            temp = np.logical_xor(indicators[l], indicators[l + 1])  # aktywowane z l na l+1
            temp1 = np.logical_or(temp, indicators[l])  # nieaktywowane z l na l+1 z wylaczeniem wczesniej aktywnych (po nalozeniu nagacji)
            activated = set()
            for i in range(data.num_users):
                for j in range(data.num_contagions):
                    if temp[i][j]:
                        if F[i][j] > 0:
                            min_pos[i] = min(min_pos[i], 1 - math.pow(1 - F[i][j], 1 / float(Y[i] + 1)))
                        else:
                            min_pos[i] = min(min_pos[i], 0)  # czy chcemy wyeliminować aktywacje, przy ujemnym wpływie?
                        activated.add(i)
                    if not temp1[i][j]:
                        max_neg[i] = max(max_neg[i], 1 - math.pow(1 - F[i][j], 1 / float(Y[i] + 1)))
            for i in activated:
                Y[i] += 1
        results = []
        for user in range(data.num_users):
            if min_pos[user] > 1:
                results.append(max(max_neg[user], 0))
            else:
                results.append(max((max_neg[user] + min_pos[user]) / 2, 0))
        # print(Results)
        # print([(i,e) for i, e in enumerate(Results) if e != 0])
        self.matrix = np.repeat(np.asarray(results)[np.newaxis].T, data.num_contagions, axis=1)
        self.initial_matrix = copy.copy(self.matrix)
        # review

    def estimate_time_batch(self, data, a_matrix, cc_matrix, volume):
        data.add_contagion_id()
        data.construct_event_log_grouped()
        indicators = []
        I = np.full((data.num_users, data.num_contagions), False, dtype=bool)
        ts = 0
        while ts < data.event_log[Data.time_stamp].max():
            for index, row in data.event_log[(data.event_log[Data.time_stamp] > ts) & (data.event_log[Data.time_stamp] <= ts + volume)].iterrows():
                I[row[Data.user]][row[Data.contagion_id]] = True
            indicators.append(I)
            I = copy.deepcopy(I)
            ts += volume
        Y = np.sum(indicators[0], axis=1)
        self._estimate(Y, a_matrix, cc_matrix, data, indicators)

    def estimate_hybride_batch(self, data):
        # TODO Implement
        pass

    # def estimateVector(self,Data):
    #     #TODO Implement
    #     indykatory_est = []
    #     I = np.full((Data.num_users_, Data.num_contagions), False, dtype=bool)
    #     for i in range(history):
    #         for index, row in event_log[event_log['ts'] == i].iterrows():
    #             I[row['userNEW'], row['tagID']] = True
    #         indykatory_est.append(I)
    #         I = copy.deepcopy(I)
    #
    # def _estimate(self,Data):
    #     #TODO Implement
    #     # Construct matrix from vector