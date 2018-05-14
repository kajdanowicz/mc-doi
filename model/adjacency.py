from collections import defaultdict
from data.data import Data

import networkx as nx
from tqdm import tqdm
import copy


class Adjacency():
    def __init__(self):
        # TODO follow sklearn way
        self.matrix = None

    def _transpose(self):
        if self.__dict__.get('matrix_transposed_',None) is None:
            self.matrix_transposed_ = self.matrix.transpose()

    def __clean_counters(self):
        # TODO What if dicts do not exist?
        del self.v_2_u_
        del self.v_and_u_
        del self.u_

    def estimate(self, data: Data):
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
