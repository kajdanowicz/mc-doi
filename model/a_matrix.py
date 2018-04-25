from collections import defaultdict

import networkx as nx
from tqdm import tqdm
import copy


class a_matrix():
    def __init__(self):
        self.matrix = None
        self.matrix_transposed = None
        self.num_users = None
        self.event_queue = dict()
        self.v_2_u = None
        self.v_and_u = None
        self.u = None

    def transpose(self):
        if self.matrix_transposed is None:
            self.matrix_transposed = self.matrix.transpose()

    def clean_counters(self):
        self.v_2_u = None
        self.v_and_u = None
        self.u = None

    def estimate(self, data):
        self.num_users = data.num_users
        self.v_2_u = defaultdict(lambda: 0)
        self.v_and_u = defaultdict(lambda: 0)
        self.u = defaultdict(lambda: 0)
        global prev_contagion
        prev_contagion = None
        data.add_graph()
        d = data.event_log.drop_duplicates(subset=['contagion', 'user'], keep='first')
        for row in d.itertuples(index=False, name=None):
            self.process_single_event(row)
            self.propagate(row[0], row[1], data.graph)
        self.reset_event_queue()
        self.calculate_weights(data.graph)
        self.clean_counters()

    def process_single_event(self, row):
        global prev_contagion
        self.u[row[1]] += 1
        if row[2] is prev_contagion:
            prev_contagion = row[2]
        else:
            self.reset_event_queue()
            prev_contagion = row[2]

    def calculate_weights(self, graph):
        G = graph.to_directed()
        nx.set_edge_attributes(G, 0, 'weight')
        for v in self.u:
            for u in G.successors(v):
                G[v][u]['weight'] = round(float(self.v_2_u[(v, u)]) / float(self.u[v]), 6)
        # for v, u in G.edges():
        #     if self.u[v] > 0:
        #         G[v][u]['weight'] = round(float(self.v_2_u[(v, u)]) / float(self.u[v]), 6)
        #     else:
        #         G[v][u]['weight'] = 0
        for i in G.nodes():
            in_degree = G.in_degree(i, weight='weight')
            if in_degree != 0:
                for v, u in G.in_edges(i):
                    G[v][u]['weight'] /= in_degree
        self.matrix = nx.adjacency_matrix(G, weight='weight')

    def reset_event_queue(self):
        self.event_queue = dict()

    def add_to_queue(self, user, ts):
        self.event_queue[user] = ts

    def propagate(self, ts, user, graph):
        for v in graph.neighbors(user):
            # if self.event_queue.get(v):
            if v in self.event_queue:
                self.v_and_u[(user, v)] += 1
                # self.v_and_u[(v, user)] += 1
                if ts - self.event_queue[v] != 0:
                    self.v_2_u[(v, user)] += 1
        self.add_to_queue(user, ts)
