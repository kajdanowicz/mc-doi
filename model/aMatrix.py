from collections import defaultdict
import networkx as nx
from tqdm import tqdm

class aMatrix():
    def __init__(self):
        self.matrix=None
        self.numUsers=None

    def estimate(self,data):
        currentContagion=None
        a_u=defaultdict(lambda: 0)
        a_v2u = defaultdict(lambda: 0)
        a_vandu = defaultdict(lambda: 0)
        tau_vu = defaultdict(lambda: 0)
        self.numUsers=data.numUsers
        data.addContagionID()
        d=data.eventLog.drop_duplicates(subset=['contagionID','user'],keep='first')
        currentTable = []
        for index, row in tqdm(d.iterrows()):
            if row['contagionID'] != currentContagion:
                currentTable = []
                currentContagion = row['contagionID']
            a_u[row['user']]+=1
            #parents=[]
            for tuple in currentTable:
                if data.edgeExists(tuple[0],row['user']) & row['ts']>tuple[1]:
                    a_v2u[(tuple[0],row['user'])]+=1
                    tau_vu[(tuple[0],row['user'])]+=(row['ts']-tuple[1])
                    #parents.append(tuple(0))
                a_vandu[(tuple[0],row['user'])]+=1
                a_vandu[(row['user'],tuple[0])]+=1
            #for v in parents:
                #update credit_v,u
            currentTable.append((row['user'],row['ts']))
        print(a_u)
        print(a_v2u)
        G=nx.from_pandas_edgelist(data.edges,'user1','user2').to_directed()
        for u,v in G.edges():
            if a_u[u]>0:
                G[u][v]['weight']=round(float(a_v2u[(u,v)]) / a_u[u],6)
            else:
                G[u][v]['weight']=0
        for u in G.nodes():
            in_degree = G.in_degree(u, weight='weight')
            if (in_degree != 0):
                for u, v in G.in_edges(u):
                    G[u][v]['weight'] /= in_degree
        self.matrix=nx.adjacency_matrix(G,weight='weight')