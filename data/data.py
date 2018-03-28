import pandas as pd
import numpy as np
from collections import defaultdict
import networkx as nx
import pickle
import itertools
import functools


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

    def loadDataFile(self, directory, fileNames = ('eventLog','edges')):
        eventLogDF = pd.read_csv(directory + fileNames[0])
        eventLogDF.columns = ['ts', 'user', 'contagion']
        edgesDF = pd.read_csv(directory + fileNames[1])
        edgesDF.columns = ['user1', 'user2']
        if data.verifyUsersCorrect(eventLogDF, edgesDF):
            self.eventLog = eventLogDF
            self.edges = edgesDF
            self.reindexUsers()
            self.addContagionID()
            self.numContagions = max(self.contagionIDDict.values()) + 1
            self.numEvents = self.eventLog.shape[0]
            self.sortData()
            return True
        else:
            return False
        # review

    def loadDataDataFrame(self, eventLogDF, edgesDF):
        eventLogDF.columns = ['ts', 'user', 'contagion']
        edgesDF.columns = ['user1', 'user2']
        if data.verifyUsersCorrect(eventLogDF, edgesDF):
            self.eventLog = eventLogDF
            self.edges = edgesDF
            self.reindexUsers()
            self.addContagionID()
            self.numContagions = max(self.contagionIDDict.values()) + 1
            self.numEvents = self.eventLog.shape[0]
            self.sortData()
            return True
        else:
            return False
        # review

    @staticmethod
    def verifyUsersCorrect(eventLogDF, edgesDF):
        if set(eventLogDF['user']).issubset(edgesDF['user1'].append(edgesDF['user2'])):
            return True
        else:
            return False
    # TODO find faster way
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
        if 'contagionID' in self.eventLog.columns:
            self.eventLog.sort_values(by=['contagionID', 'ts'], inplace=True)
        else:
            self.eventLog.sort_values(by=['contagion', 'ts'], inplace=True)

    def loadData(self, directory=None, eventLogDF=None, edgesDF=None, fileNames = ('eventLog','edges')):
        ''' Loads data to class data instance from the source that depends on given arguments'''
        if directory is not None:
            if not self.loadDataFile(directory, fileNames):
                return False
        elif (eventLogDF is not None) and (edgesDF is not None):
            if not self.loadDataDataFrame(eventLogDF, edgesDF):
                return False
        else:
            return False
        return True
        # review

    def edgeExists(self, user1, user2):
        if self.edges[(self.edges['user1'].isin([user1, user2])) & (self.edges['user2'].isin([user1, user2]))].empty:
            return False
        else:
            return True

    def loadDataMinOccurrence(self, minOccurs, directory=None, eventLogDF=None, edgesDF=None):
        """ Loads data to class data instance from the source that depends on given arguments
        Only contagions appearing in minOccurs events are loaded"""
        if not self.loadData(directory, eventLogDF, edgesDF):
            return False
        self.restrictEventLogMinOccurences(minOccurs)
        return True
        # review

    def restrictEventLogMinOccurences(self, minOccurs, maxNumContagions = None):
        """ Restricts events in self to that, which contains contagions appearing in the data minOccurs times."""
        # TODO Use deleteContagions to obtain this
        temp = self.eventLog.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        if maxNumContagions is None:
            series = temp[(temp['ts'] >= minOccurs)]['contagion']
            self.eventLog = self.eventLog[self.eventLog['contagion'].isin(series)]
            self.updateEventLog()
        else:
            series = temp[(minOccurs <= temp['ts'])].sort_values(by='ts').iloc[:maxNumContagions]['contagion']
            self.eventLog = self.eventLog[self.eventLog['contagion'].isin(series)]
            self.updateEventLog()
        # review

    def restrictEventLogMaxOccurences(self, maxOccurs, maxNumContagions = None):
        """ Restricts events in self to that, which contains contagions appearing in the data minOccurs times."""
        # TODO Use deleteContagions to obtain this
        temp = self.eventLog.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        if maxNumContagions is None:
            series = temp[(temp['ts'] <= maxOccurs)]['contagion']
            self.eventLog = self.eventLog[self.eventLog['contagion'].isin(series)]
            self.updateEventLog()
        else:
            series = temp[(temp['ts'] <= maxOccurs)].sort_values(by='ts', ascending=False).iloc[:maxNumContagions]['contagion']
            self.eventLog = self.eventLog[self.eventLog['contagion'].isin(series)]
            self.updateEventLog()
        # review

    def restrictEventLogMinMaxOccurences(self,minOccurs,maxOccurs):
        temp = self.eventLog.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        series = temp[(minOccurs <= temp['ts']) & (temp['ts'] <= maxOccurs)]['contagion']
        self.eventLog = self.eventLog[self.eventLog['contagion'].isin(series)]
        self.updateEventLog()

    def restrictEventLogMaxNumContagions(self, maxNumContagions):
        temp = self.eventLog.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        series = temp.sort_values(by='ts', ascending=False).iloc[:maxNumContagions]['contagion']
        self.eventLog = self.eventLog[self.eventLog['contagion'].isin(series)]
        self.updateEventLog()

    def restrictEventLog(self, maxOccurs = None, minOccurs = None, maxNumContagions = None):
        """ """
        # TODO Use deleteContagions to obtain this
        if (maxOccurs is not None) and (minOccurs is None):
            self.restrictEventLogMaxOccurences(maxOccurs,maxNumContagions)
        elif (maxOccurs is not None) and (minOccurs is not None) and (minOccurs<=maxOccurs):
            self.restrictEventLogMinMaxOccurences(minOccurs,maxOccurs)
        elif (maxOccurs is None) and (minOccurs is not None):
            self.restrictEventLogMinOccurences(minOccurs,maxNumContagions)
        elif (maxOccurs is None) and (minOccurs is None) and (maxNumContagions is not None):
            self.restrictEventLogMaxNumContagions(maxNumContagions)
        # review


    def deleteUsers(self, userList):
        self.edges.drop(self.edges[(self.edges['user1'].isin(userList)) | (self.edges['user2'].isin(userList))].index,
                        inplace=True)
        self.eventLog.drop(self.eventLog[self.eventLog['user'].isin(userList)].index, inplace=True)
        self.updateEventLog()
        self.removeFromGraph(userList)
        self.reindexUsers()
        # review

    def keepUsers(self, userList):
        '''
        "private" method
        Does not check if contagions have at least one event
        '''
        row_mask = self.edges.isin(userList).any(1)
        self.edges = self.edges[row_mask]
        # self.eventLog = self.eventLog[self.eventLog['user'].isin(userList)]
        # self.numEvents = self.eventLog.shape[0]

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
            t = defaultdict(functools.partial(next, itertools.count()))
            self.eventLog = self.eventLog.assign(contagionID=self.eventLog['contagion'].map(t))
            self.contagionIDDict = t
        else:
            pass
        # review

    def constructEventLogGrouped(self):
        if 'eventID' not in self.eventLog.columns:
            t = defaultdict(functools.partial(next, itertools.count()))
            self.eventLog = self.eventLog.assign(eventID=self.eventLog.apply(lambda row: t[(row['user'], row['ts'])], axis=1, reduce = True))
        # review

    def toCSV(self, directory=''):
        self.eventLog.to_csv(directory + 'eventLog', header=False, index=False)
        self.edges.to_csv(directory + 'edges', header=False, index=False)

    def sampleEvents(self,fraction):
        self.eventLog=self.eventLog.sample(frac=fraction)
        self.updateEventLog()
        self.numEvents = self.eventLog.shape[0]

    def sampleEdges(self,fraction,number=None):
        if number is None:
            self.edges = self.edges.sample(frac=fraction)
        else:
            self.edges = self.edges.sample(frac=float(number)/self.edges.shape[0])
        self.keepEventsOfUsers(set(self.edges.user1.unique()).union(self.edges.user2.unique()))
        self.reindexUsers()
        self.updateEventLog()

    def toPickle(self,directory=''):
        pickle.dump(self, open(directory + 'data.p', 'wb'))

    def restrictUsersToActive(self):
        activeUsers = self.eventLog.user.unique()
        self.keepUsers(activeUsers)
        if self.graph is not None:
            self.restrictGraph(activeUsers)
        self.reindexUsers()

    def restrictUsersTo(self, userList):
        self.keepUsers(userList)
        if self.graph is not None:
            self.restrictGraph(userList)
        self.keepEventsOfUsers(userList)
        self.reindexUsers()

    def deleteEventsOfUsers(self,userList):
        self.eventLog.drop(self.eventLog[self.eventLog['user'].isin(userList)].index, inplace=True)

    def keepEventsOfUsers(self,userList):
        self.eventLog = self.eventLog[self.eventLog.user.isin(userList)]

    def restrictGraph(self,userList):
        self.graph.remove_nodes_from(np.setdiff1d(self.graph.nodes(),userList))

    def removeFromGraph(self,userList):
        self.graph.remove_nodes_from(userList)

    def reindexUsersInEventLog(self,dictionary):
        self.eventLog.user = self.eventLog.user.map(dictionary)

    def reindexContagionID(self,dictionary):
        self.eventLog = self.eventLog.assign(contagionID=self.eventLog['contagion'].map(dictionary))

    def updateEventLog(self):
        if 'contagionID' in self.eventLog.columns:
            t = defaultdict(functools.partial(next, itertools.count()))
            self.reindexContagionID(t)
            self.numContagions = max(t.values()) + 1
            self.contagionIDDict = t
        else:
            self.numContagions = len(self.eventLog.contagion.unique())
        self.numEvents = self.eventLog.shape[0]

    def reindexUsersInGraph(self,dictionary):
        self.graph = nx.relabel_nodes(self.graph,dictionary,copy=True)

    def reindexUsers(self):
        t = defaultdict(functools.partial(next, itertools.count()))
        self.edges.user1 = self.edges.user1.map(t)
        self.edges.user2 = self.edges.user2.map(t)
        self.reindexUsersInEventLog(t)
        if self.graph is not None:
            self.reindexUsersInGraph(t)
        self.numUsers = max(t.values())+1

    def prepareTestData(self,fraction):
        minTime = self.eventLog.ts.min()
        maxTime = self.eventLog.ts.max()
        stoppingTime = float(maxTime-minTime)*fraction + minTime
        tmp = self.eventLog[self.eventLog.ts > stoppingTime]
        self.eventLog = self.eventLog[self.eventLog.ts <= stoppingTime]
        self.numEvents = self.eventLog.shape[0]
        return tmp

    @staticmethod
    def fromPickle(directory):
        return pickle.load(open(directory+'data.p','rb'))
