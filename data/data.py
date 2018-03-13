import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import pickle


class data():

    def __init__(self):
        self.eventLog = None
        self.edges = None
        self.numUsers = None
        self.numContagions = None
        self.numEvents = None
        self.contagionIDDict = None
        self.graph = None

    def addGraph(self):
        if self.graph is None:
            self.graph = nx.from_pandas_edgelist(self.edges, 'user1', 'user2')

    def loadDataFile(self, directory):
        eventLogDF = pd.read_csv(directory + 'eventLog')
        eventLogDF.columns = ['ts', 'user', 'contagion']
        edgesDF = pd.read_csv(directory + 'edges')
        edgesDF.columns = ['user1', 'user2']
        if data.verifyUsersCorrect(eventLogDF, edgesDF):
            self.eventLog = eventLogDF
            self.edges = edgesDF
            self.sortData()
            self.numUsers = len(np.union1d(self.edges['user1'], self.edges['user2']))
            self.numContagions = len(self.eventLog['contagion'].unique())
            self.numEvents = self.eventLog.shape[0]
            return True
        else:
            return False
        # TODO Implement Exception
        # review

    def loadDataDataFrame(self, eventLogDF, edgesDF):
        eventLogDF.columns = ['ts', 'user', 'contagion']
        edgesDF.columns = ['user1', 'user2']
        if data.verifyUsersCorrect(eventLogDF, edgesDF):
            self.eventLog = eventLogDF
            self.edges = edgesDF
            self.sortData()
            self.numUsers = len(np.union1d(self.edges['user1'], self.edges['user2']))
            self.numContagions = len(self.eventLog['contagion'].unique())
            self.numEvents = self.eventLog.shape[0]
            return True
        else:
            return False
        # TODO Implement Exception
        # review

    @staticmethod
    def verifyUsersCorrect(eventLogDF, edgesDF):
        if np.setdiff1d(eventLogDF['user'], np.union1d(edgesDF['user1'], edgesDF['user2'])).shape[0] == 0:
            return True
        else:
            return False

    # review

    def verifyDataCorrect(self):
        if not data.verifyUsersCorrect(self.eventLog, self.edges):
            return False
        elif not self.numEvents == self.eventLog.shape[0]:
            return False
        elif not self.numContagions == len(self.eventLog['contagions'].unique()):
            return False
        elif not self.numUsers == len(np.union1d(self.edges['user1'], self.edges['user2'])):
            return False
        else:
            return True

    # review
    def sortData(self):
        self.eventLog.sort_values(by=['contagion', 'ts'], inplace=True)

    def loadData(self, directory=None, eventLogDF=None, edgesDF=None):
        ''' Loads data to class data instance from the source that depends on given arguments'''
        if directory is not None:
            if self.loadDataFile(directory):
                u = defaultdict(lambda: len(u))
                self.edges['user1'] = self.edges.apply(lambda row: u[row['user1']], axis=1)
                self.edges['user2'] = self.edges.apply(lambda row: u[row['user2']], axis=1)
                self.eventLog['user'] = self.eventLog['user'].map(u)
                return True
            else:
                return False
        elif (eventLogDF is not None) and (edgesDF is not None):
            if self.loadDataDataFrame(eventLogDF, edgesDF):
                u = defaultdict(lambda: len(u))
                self.edges['user1'] = self.edges.apply(lambda row: u[row['user1']], axis=1)
                self.edges['user2'] = self.edges.apply(lambda row: u[row['user2']], axis=1)
                self.eventLog['user'] = self.eventLog['user'].map(u)
                return True
            else:
                return False
        else:
            return False
        # review

    def edgeExists(self, user1, user2):
        if self.edges[(self.edges['user1'].isin([user1, user2])) & (self.edges['user2'].isin([user1, user2]))].empty:
            return False
        else:
            return True

    def loadDataMinOccurrence(self, minOccurs, directory=None, eventLogDF=None, edgesDF=None):
        """ Loads data to class data instance from the source that depends on given arguments
        Only contagions appearing in minOccurs events are loaded"""
        if self.loadData(directory, eventLogDF, edgesDF) == False:
            return False
        self.restrictEventLogMinOccurences(minOccurs)
        return True
        # review

    def restrictEventLogMinOccurences(self, minOccurs):
        """ Restricts events in self to that, which contains contagions appearing in the data minOccurs times."""
        # TODO Use deleteContagions to obtain this
        temp = self.eventLog.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        series = temp[(temp['ts'] > minOccurs)]['contagion']
        self.eventLog = self.eventLog[self.eventLog['contagion'].isin(series)]
        self.numContagions = len(series)
        self.numEvents = self.eventLog.shape[0]
        # t = defaultdict(lambda: len(t))
        # temp['tagID'] = temp.apply(lambda row: t[row['contagion']], axis=1)
        # review

    def deleteUsers(self, userList):
        self.edges.drop(self.edges[(self.edges['user1'].isin(userList)) | (self.edges['user2'].isin(userList))].index,
                        inplace=True)
        self.eventLog.drop(self.eventLog[self.eventLog['user'].isin(userList)].index, inplace=True)
        # pd.concat([self.edges['user1'],self.edges['user2']],copy=False).apply(lambda row: u[row.loc[:,0]])
        u = defaultdict(lambda: len(u))
        self.edges['user1'] = self.edges.apply(lambda row: u[row['user1']], axis=1)
        self.edges['user2'] = self.edges.apply(lambda row: u[row['user2']], axis=1)
        self.eventLog['user'] = self.eventLog['user'].map(u)
        self.numUsers = len(np.union1d(self.edges['user1'], self.edges['user2']))
        self.numEvents = self.eventLog.shape[0]
        self.numContagions = len(self.eventLog['contagion'].unique())
        # review

    def dropEdge(self, edge=None, user1=None, user2=None):
        if (user1 is None) & (user2 is None):
            self.edges.drop(self.edges[((self.edges['user1'] == edge[0]) & (self.edges['user2'] == edge[1])) | (
                    (self.edges['user1'] == edge[1]) & (self.edges['user2'] == edge[0]))].index, inplace=True)
        elif (edge is None) & (user1 is not None) & (user2 is not None):
            self.edges.drop(self.edges[(self.edges['user1'].isin([user1, user2])) & (
                self.edges['user2'].isin([user1, user2]))].index, inplace=True)
        else:
            return False
        # question Should an unconnected user be deleted?
        # review

    def deleteEdges(self, edgesList):
        # question Should an unconnected user be deleted?
        for edge in edgesList:
            self.dropEdge(edge)
        # review

    def deleteContagions(self, contagionList):
        self.eventLog.drop(self.eventLog[self.eventLog['contagion'].isin(contagionList)].index, inplace=True)
        self.numEvents = self.eventLog.shape[0]
        self.numContagions = len(self.eventLog['contagion'].unique())
        # review

    def deleteContagionsByID(self, contagionIDList):
        if 'contagionID' in self.eventLog.columns:
            self.eventLog.drop(self.eventLog[self.eventLog['contagionID'].isin(contagionIDList)].index, inplace=True)
            self.numEvents = self.eventLog.shape[0]
            self.numContagions = len(self.eventLog['contagionID'].unique())
            # review

    def addContagionID(self):
        if 'contagionID' not in self.eventLog.columns:
            t = defaultdict(lambda: len(t))
            self.eventLog['contagionID'] = self.eventLog.apply(lambda row: t[row['contagion']], axis=1)
            self.contagionIDDict = t
        # TODO Does it work globally?
        # review

    def constructEventLogGrouped(self):
        t = defaultdict(lambda: len(t))
        self.eventLog['eventID'] = self.eventLog.apply(lambda row: t[(row['user'], row['ts'])], axis=1)
        # review

    def toCSV(self, directory=''):
        self.eventLog.to_csv(directory + 'eventLog', header=False, index=False)
        self.edges.to_csv(directory + 'edges', header=False, index=False)

    def sample(self,fraction):
        self.eventLog=self.eventLog.sample(frac=fraction)
        self.numContagions = len(self.eventLog['contagion'].unique())
        self.numEvents = self.eventLog.shape[0]

    def toPickle(self,directory):
        pickle.dump(self, open(directory + 'data.p', 'wb'))

    @staticmethod
    def fromPickle(directory):
        return pickle.load(open(directory+'data.p','rb'))
