import copy
import math
from abc import abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from numpy import ndarray

from data.data import Data


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
    # TODO Finish docstring - describe attributes
    """
    Class for estimation and storing adjacency matrix of multi-contagion threshold model.
    """
    def __init__(self):
        super(Adjacency, self).__init__()

    def estimate(self, data: Data, **kwargs):
        """
        Estimates edge weights in Linear Threshold model setting according to [1]_. It uses Bernoulli trial MLE or
        Jaccard index depending on keyword argument :name:`function`, default Bernoulli.

        Parameters
        ----------
        data : Data
            :class:`Data` object according to which adjacency matrix should be estimated.
        kwargs
            Arbitrary keyword arguments.


            .. [1] Goyal, A., Bonchi, F., & Lakshmanan, L. V. S. (2010). Learning influence probabilities in social
                networks. In Proceedings of the third ACM international conference on Web search and data mining - WSDM ’10 (
                p. 241).

        """
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
        function = kwargs.get('function','bernoulli')
        edge_probability_func = {'bernoulli': self.__MLE_Bernoulli_trial, 'jaccard' : self.__jaccard_index}
        self._calculate_weights(data.graph, edge_probability_func[function])
        self.__clean_counters()
        # review function

    def assign_matrix(self, matrix):
        #TODO Implement this method
        pass

    def transpose(self):
        """
        Creates :name:`matrix_transposed_` attribute containing transposition of :name:`matrix`.
        """
        if self.__dict__.get('matrix_transposed_',None) is None:
            self.matrix_transposed_ = self.matrix.transpose()

    def __clean_counters(self):
        if self.__dict__.get('v_2_u_',None) is None:
            raise NameError('Can not delete self.v_2_u_, it does not exist')
        else:
            del self.v_2_u_
        if self.__dict__.get('v_and_u_',None) is None:
            raise NameError('Can not delete self.v_and_u_, it does not exist')
        else:
            del self.v_and_u_
        if self.__dict__.get('u_',None) is None:
            raise NameError('Can not delete self.u_, it does not exist')
        else:
            del self.u_

    def __verify_contagion(self, row: tuple, prev_contagion: int) -> int:
        self.u_[row[1]] += 1
        if row[2] is not prev_contagion:
            self.__reset_event_queue()
            prev_contagion = row[2]
        return prev_contagion

    def _calculate_weights(self, graph, edge_probability_func):
        G = graph.to_directed()
        nx.set_edge_attributes(G, 0, 'weight')
        for v in self.u_.keys():
            for u in G.successors(v):
                G[v][u]['weight'] = edge_probability_func(v, u)
        for i in G.nodes():
            in_degree = G.in_degree(i, weight='weight')
            if in_degree != 0:
                for v, u in G.in_edges(i):
                    G[v][u]['weight'] /= in_degree
        self.matrix = nx.adjacency_matrix(G, weight='weight')

    def __MLE_Bernoulli_trial(self, v: int, u: int) -> float:
        """
        Computes (v,u) edge probability in sense of the Independent Cascade Model using Maximum Likelihood Estimator
        for Bernoulli distribution.

        Parameters
        ----------
        v : int
            Source node.
        u : int
            Target node.

        Returns
        -------
        float
            Calculated edge probability in sense of the Independent Cascade Model.


        According to [1]_ in Bernoulli distribution static model:

        .. math:: p_{v,u} = \frac{A_{v2u } }{A_v },

        where :math:`A_{v2u}` is the number of actions propagate from :math:`v` to :math:`u` and :math:`A_{v}` is the
        number of actions performed by :math:`v`.

        .. [1] Goyal, A., Bonchi, F., & Lakshmanan, L. V. S. (2010). Learning influence probabilities in social
            networks. In Proceedings of the third ACM international conference on Web search and data mining - WSDM ’10 (
            p. 241).

        """
        return round(float(self.v_2_u_[(v, u)]) / float(self.u_[v]), 6)

    def __jaccard_index(self, v, u):
        """
        Computes (v,u) edge probability in sense of the Independent Cascade Model using Jaccard index.

        Parameters
        ----------
        v : int
            Source node.
        u : int
            Target node.

        Returns
        -------
        float
            Calculated edge probability in sense of the Independent Cascade Model.


        According to [1]_ in Jaccard index static model:

        .. math:: p_{v,u} = \frac{A_{v2u } }{A{u|v} },

        where :math:`A_{v2u}` is the number of actions propagate from :math:`v` to :math:`u` and :math:`A_{u|v}` is the
        number of actions performed either by :math:`u` or :math:`v`.

        .. [1] Goyal, A., Bonchi, F., & Lakshmanan, L. V. S. (2010). Learning influence probabilities in social
            networks. In Proceedings of the third ACM international conference on Web search and data mining - WSDM ’10 (
            p. 241).

        """
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
        # TODO write docstring
        """
        """
        # TODO Reconsider sparse matrix implementation
        # TODO Consider using PyTables
        data.add_contagion_id()
        self.num_contagions=data.num_contagions
        self.matrix = np.eye(N=self.num_contagions)
        self.num_users_performing_events=len(data.event_log.user.unique())
        unique_event_log = data.event_log[[Data.user, Data.contagion_id]].drop_duplicates(subset=None, keep='first', inplace=False)
        co_occurrence_counter = pd.merge(unique_event_log[[Data.user, Data.contagion_id]], unique_event_log[[Data.user, Data.contagion_id]], on=Data.user,suffixes=('_1','_2')).groupby([Data.contagion_id+'_1',Data.contagion_id+'_2']).count()
        for i in range(self.num_contagions):
            count_i = float(co_occurrence_counter.loc[(i, i)].values[0])
            for j in range(i + 1, self.num_contagions):
                count_j = float(co_occurrence_counter.loc[(j, j)].values[0])
                if (i,j) in co_occurrence_counter.index:
                    count_ij = float(co_occurrence_counter.loc[(i, j)].values[0])
                else:
                    count_ij = 0.
                contagion_correlation = count_ij / math.sqrt(count_i * count_j) - ((count_j-count_ij) / math.sqrt((self.num_users_performing_events - count_i) * count_j) + (count_i - count_ij) / math.sqrt(count_i * (self.num_users_performing_events - count_j))) / 2
                self.matrix[i][j] = contagion_correlation
                self.matrix[j][i] = contagion_correlation

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

    def estimate(self, data: Data, *args, **kwargs):
        batch_type = kwargs.get('batch_type', None)
        adjacency_matrix = args[0]
        correlation_matrix = args[1]
        batch_size = kwargs.get('batch_size')
        if batch_type == 'volume':
            self.estimate_volume_batch(data, adjacency_matrix, correlation_matrix, batch_size)
        elif batch_type == 'time':
            self.estimate_time_batch(data, adjacency_matrix, correlation_matrix, batch_size)
        else:
            raise NameError('Can not estimate thresholds. No such method as '+batch_type)

    def assign_matrix(self, matrix):
        #TODO Implement this method
        pass

    def estimate_volume_batch(self, data, a_matrix, cc_matrix, batch_size):
        data.add_contagion_id()
        data.construct_event_log_grouped()
        indicators = []
        I = np.full((data.num_users, data.num_contagions), False, dtype=bool)
        event_id = 0
        while event_id < data.event_log[Data.event_id].max():
            for index, row in data.event_log[(data.event_log[Data.event_id] > event_id) & (data.event_log[Data.event_id] <= event_id + batch_size)].iterrows():
                I[row[Data.user]][row[Data.contagion_id]] = True
            indicators.append(I)
            I = copy.deepcopy(I)
            event_id += batch_size
        Y = np.sum(indicators[0], axis=1)
        self._estimate(Y, a_matrix, cc_matrix, data, indicators)

    def estimate_time_batch(self, data, a_matrix, cc_matrix, batch_size):
        data.add_contagion_id()
        data.construct_event_log_grouped()
        indicators = []
        I = np.full((data.num_users, data.num_contagions), False, dtype=bool)
        ts = 0
        while ts < data.event_log[Data.time_stamp].max():
            for index, row in data.event_log[(data.event_log[Data.time_stamp] > ts) & (data.event_log[Data.time_stamp] <= ts + batch_size)].iterrows():
                I[row[Data.user]][row[Data.contagion_id]] = True
            indicators.append(I)
            I = copy.deepcopy(I)
            ts += batch_size
        Y = np.sum(indicators[0], axis=1)
        self._estimate(Y, a_matrix, cc_matrix, data, indicators)

    def _estimate(self, Y, a_matrix: Adjacency, cc_matrix, data, indicators):
        # TODO refactor
        a_matrix.transpose()
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

    def estimate_hybrid_batch(self, data):
        # TODO Implement
        pass