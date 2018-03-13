from collections import defaultdict

import networkx as nx
from tqdm import tqdm
import copy

class aMatrix():
    def __init__(self):
        self.matrix = None
        self.matrixTransposed = None
        self.numUsers = None
        self.eventQueue = dict()
        self.v2u = None
        self.vANDu = None
        self.u = None

    def transpose(self):
        if self.matrixTransposed is None:
            self.matrixTransposed = self.matrix.transpose()

    def cleanCounters(self):
        self.v2u = None
        self.vANDu = None
        self.u = None

    def estimate(self,data):
        self.numUsers=data.numUsers
        self.v2u=defaultdict(lambda: 0)
        self.vANDu=defaultdict(lambda: 0)
        self.u=defaultdict(lambda: 0)
        prevContagion=None
        data.addGraph()
        d = data.eventLog.drop_duplicates(subset=['contagion', 'user'], keep='first')
        for row in tqdm(d.itertuples(index=False,name=None)):
            self.u[row[1]]+=1
            if(row[2] is prevContagion):
                prevContagion=row[2]
            else:
                self.resetEventQueue()
                prevContagion=row[2]
            self.propagate(row[0],row[1],data.graph)
        self.resetEventQueue()
        self.calculateWeights(data.graph)
        self.cleanCounters()

    def calculateWeights(self,graph):
        G = graph.to_directed()
        nx.set_edge_attributes(G, 0, 'weight')
        for v in self.u:
            for u in G.successors(v):
                G[v][u]['weight'] = round(float(self.v2u[(v, u)]) / float(self.u[v]), 6)
        # for v, u in G.edges():
        #     if self.u[v] > 0:
        #         G[v][u]['weight'] = round(float(self.v2u[(v, u)]) / float(self.u[v]), 6)
        #     else:
        #         G[v][u]['weight'] = 0
        for i in G.nodes():
            in_degree = G.in_degree(i, weight='weight')
            if in_degree != 0:
                for v, u in G.in_edges(i):
                    G[v][u]['weight'] /= in_degree
        self.matrix = nx.adjacency_matrix(G, weight='weight')



    def resetEventQueue(self):
        self.eventQueue = dict()

    def addToQueue(self,user,ts):
        self.eventQueue[user]=ts

    def propagate(self, ts, user, graph):
        for v in graph.neighbors(user):
            #if self.eventQueue.get(v):
            if v in self.eventQueue:
                self.vANDu[(user,v)] += 1
                #self.vANDu[(v, user)] += 1
                if ts-self.eventQueue[v] != 0:
                    self.v2u[(v,user)] += 1
        self.addToQueue(user,ts)
