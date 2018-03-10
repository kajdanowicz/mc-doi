from collections import defaultdict

import networkx as nx
from tqdm import tqdm
import copy

class aMatrix():
    def __init__(self):
        self.matrix = None
        self.numUsers = None
        self.eventQueue = dict()
        self.v2u = None
        self.vANDu = None
        self.u = None

    def cleanCounters(self):
        self.v2u = None
        self.vANDu = None
        self.u = None

    def estimate1(self,data):
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



    def estimate(self, data):
        currentContagion = None
        a_u = defaultdict(lambda: 0)
        a_v2u = defaultdict(lambda: 0)
        a_vandu = defaultdict(lambda: 0)
        tau_vu = defaultdict(lambda: 0)
        self.numUsers = data.numUsers
        # data.addContagionID()
        d = data.eventLog.drop_duplicates(subset=['contagion', 'user'], keep='first')
        data.addGraph()
        currentTable = []
        for index, row in tqdm(d.iterrows()):
            user_id = copy.copy(row['user'])
            ts=copy.copy(row['ts'])
            if row['contagion'] != currentContagion:
                currentTable = []
                currentContagion = copy.copy(row['contagion'])
            a_u[user_id] += 1
            # parents=[]
            # TODO Iterowac po sasiadach
            for tuple in currentTable:
                if (data.graph[user_id].get(tuple[0]) is not None) & ts > tuple[1]:
                    a_v2u[(tuple[0], user_id)] += 1
                    tau_vu[(tuple[0], user_id)] += (row['ts'] - tuple[1])
                    # parents.append(tuple(0))
                a_vandu[(tuple[0], user_id)] += 1
                a_vandu[(user_id, tuple[0])] += 1
            # for v in parents:
            # update credit_v,u
            currentTable.append((user_id, ts))
        # print(a_u)
        # print(a_v2u)
        G = data.graph.to_directed()
        for u, v in G.edges():
            if a_u[u] > 0:
                G[u][v]['weight'] = round(float(a_v2u[(u, v)]) / a_u[u], 6)
            else:
                G[u][v]['weight'] = 0
        for u in G.nodes():
            in_degree = G.in_degree(u, weight='weight')
            if (in_degree != 0):
                for u, v in G.in_edges(u):
                    G[u][v]['weight'] /= in_degree
        self.matrix = nx.adjacency_matrix(G, weight='weight')
