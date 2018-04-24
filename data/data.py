import functools
import itertools
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


class Data:
    def __init__(self, event_log, ):

        self.event_log = None
        self.edges = None
        self.num_users = None
        self.num_contagions = None
        self.num_events = None
        self.contagion_id_dict = None
        self.graph = None

    def add_graph(self):
        """
        Method adds Networkx graph to data object
        :return:
        """
        if self.graph is None:
            self.graph = nx.from_pandas_edgelist(self.edges, 'user1', 'user2')

    def load_data_file(self, directory, fileNames=('eventLog', 'edges')):
        """

        :param directory:
        :param fileNames:
        :return:
        """
        eventLogDF = pd.read_csv(directory + fileNames[0])
        eventLogDF.columns = ['ts', 'user', 'contagion']
        edgesDF = pd.read_csv(directory + fileNames[1])
        edgesDF.columns = ['user1', 'user2']
        if Data.verify_users_correct(eventLogDF, edgesDF):
            self.event_log = eventLogDF
            self.edges = edgesDF
            self.reindexUsers()
            self.addContagionID()
            self.num_contagions = max(self.contagion_id_dict.values()) + 1
            self.num_events = self.event_log.shape[0]
            self.sortData()
            return True
        else:
            return False
            # review

    def load_data_data_frame(self, eventLogDF, edgesDF):
        """

        :param pd.DataFrame eventLogDF: Data frame containing event log
        :param pd.DataFrame edgesDF: Data frame containing
        :return: If data have been loaded successfully
        :rtype: bool
        """
        eventLogDF.columns = ['ts', 'user', 'contagion']
        edgesDF.columns = ['user1', 'user2']
        if Data.verify_users_correct(eventLogDF, edgesDF):
            self.event_log = eventLogDF
            self.edges = edgesDF
            self.reindexUsers()
            self.addContagionID()
            self.num_contagions = max(self.contagion_id_dict.values()) + 1
            self.num_events = self.event_log.shape[0]
            self.sortData()
            return True
        else:
            return False
            # review

    @staticmethod
    def verify_users_correct(event_log_df, edges_df):
        if set(event_log_df['user']).issubset(edges_df['user1'].append(edges_df['user2'])):
            return True
        else:
            return False

    # TODO find faster way
    # review

    def verify_data_correct(self):
        if not Data.verify_users_correct(self.event_log, self.edges):
            return False
        elif not self.num_events == self.event_log.shape[0]:
            return False
        elif not self.num_contagions == len(self.event_log['contagions'].unique()):
            return False
        elif not self.num_users == len(np.union1d(self.edges['user1'], self.edges['user2'])):
            return False
        else:
            return True

    # review

    def sortData(self):
        if 'contagionID' in self.event_log.columns:
            self.event_log.sort_values(by=['contagionID', 'ts'], inplace=True)
        else:
            self.event_log.sort_values(by=['contagion', 'ts'], inplace=True)

    def loadData(self, directory=None, eventLogDF=None, edgesDF=None, fileNames=('eventLog', 'edges')):
        ''' Loads data to class data instance from the source that depends on given arguments'''
        if directory is not None:
            if not self.load_data_file(directory, fileNames):
                return False
        elif (eventLogDF is not None) and (edgesDF is not None):
            if not self.load_data_data_frame(eventLogDF, edgesDF):
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

    def restrictEventLogMinOccurences(self, minOccurs, maxNumContagions=None):
        """ Restricts events in self to that, which contains contagions appearing in the data minOccurs times."""
        # TODO Use deleteContagions to obtain this
        temp = self.event_log.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        if maxNumContagions is None:
            series = temp[(temp['ts'] >= minOccurs)]['contagion']
            self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
            self.updateEventLog()
        else:
            series = temp[(minOccurs <= temp['ts'])].sort_values(by='ts').iloc[:maxNumContagions]['contagion']
            self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
            self.updateEventLog()
            # review

    def restrictEventLogMaxOccurences(self, maxOccurs, maxNumContagions=None):
        """ Restricts events in self to that, which contains contagions appearing in the data minOccurs times."""
        # TODO Use deleteContagions to obtain this
        temp = self.event_log.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        if maxNumContagions is None:
            series = temp[(temp['ts'] <= maxOccurs)]['contagion']
            self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
            self.updateEventLog()
        else:
            series = temp[(temp['ts'] <= maxOccurs)].sort_values(by='ts', ascending=False).iloc[:maxNumContagions][
                'contagion']
            self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
            self.updateEventLog()
            # review

    def restrictEventLogMinMaxOccurences(self, minOccurs, maxOccurs):
        temp = self.event_log.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        series = temp[(minOccurs <= temp['ts']) & (temp['ts'] <= maxOccurs)]['contagion']
        self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
        self.updateEventLog()

    def restrictEventLogMaxNumContagions(self, maxNumContagions):
        temp = self.event_log.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        series = temp.sort_values(by='ts', ascending=False).iloc[:maxNumContagions]['contagion']
        self.event_log = self.event_log[self.event_log['contagion'].isin(series)]
        self.updateEventLog()

    def restrictEventLog(self, maxOccurs=None, minOccurs=None, maxNumContagions=None):
        """ """
        # TODO Use deleteContagions to obtain this
        if (maxOccurs is not None) and (minOccurs is None):
            self.restrictEventLogMaxOccurences(maxOccurs, maxNumContagions)
        elif (maxOccurs is not None) and (minOccurs is not None) and (minOccurs <= maxOccurs):
            self.restrictEventLogMinMaxOccurences(minOccurs, maxOccurs)
        elif (maxOccurs is None) and (minOccurs is not None):
            self.restrictEventLogMinOccurences(minOccurs, maxNumContagions)
        elif (maxOccurs is None) and (minOccurs is None) and (maxNumContagions is not None):
            self.restrictEventLogMaxNumContagions(maxNumContagions)
            # review

    def deleteUsers(self, userList):
        self.edges.drop(self.edges[(self.edges['user1'].isin(userList)) | (self.edges['user2'].isin(userList))].index,
                        inplace=True)
        self.event_log.drop(self.event_log[self.event_log['user'].isin(userList)].index, inplace=True)
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
        self.event_log.drop(self.event_log[self.event_log['contagion'].isin(contagionList)].index, inplace=True)
        self.num_events = self.event_log.shape[0]
        self.num_contagions = len(self.event_log['contagion'].unique())
        # review

    def deleteContagionsByID(self, contagionIDList):
        if 'contagionID' in self.event_log.columns:
            self.event_log.drop(self.event_log[self.event_log['contagionID'].isin(contagionIDList)].index, inplace=True)
            self.num_events = self.event_log.shape[0]
            self.num_contagions = len(self.event_log['contagionID'].unique())
            # review

    def addContagionID(self):
        if 'contagionID' not in self.event_log.columns:
            t = defaultdict(functools.partial(next, itertools.count()))
            self.event_log = self.event_log.assign(contagionID=self.event_log['contagion'].map(t))
            self.contagion_id_dict = t
        else:
            pass
            # review

    def constructEventLogGrouped(self):
        if 'eventID' not in self.event_log.columns:
            t = defaultdict(functools.partial(next, itertools.count()))
            self.event_log = self.event_log.assign(
                eventID=self.event_log.apply(lambda row: t[(row['user'], row['ts'])], axis=1, reduce=True))
            # review

    def toCSV(self, directory=''):
        self.event_log.to_csv(directory + 'eventLog', header=False, index=False)
        self.edges.to_csv(directory + 'edges', header=False, index=False)

    def sampleEvents(self, fraction):
        self.event_log = self.event_log.sample(frac=fraction)
        self.updateEventLog()
        self.num_events = self.event_log.shape[0]

    def sampleEdges(self, fraction, number=None):
        if number is None:
            self.edges = self.edges.sample(frac=fraction)
        else:
            self.edges = self.edges.sample(frac=float(number) / self.edges.shape[0])
        self.keepEventsOfUsers(set(self.edges.user1.unique()).union(self.edges.user2.unique()))
        self.reindexUsers()
        self.updateEventLog()

    def toPickle(self, directory=''):
        pickle.dump(self, open(directory + 'data.p', 'wb'))

    def restrictUsersToActive(self):
        activeUsers = self.event_log.user.unique()
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

    def deleteEventsOfUsers(self, userList):
        self.event_log.drop(self.event_log[self.event_log['user'].isin(userList)].index, inplace=True)

    def keepEventsOfUsers(self, userList):
        self.event_log = self.event_log[self.event_log.user.isin(userList)]

    def restrictGraph(self, userList):
        self.graph.remove_nodes_from(np.setdiff1d(self.graph.nodes(), userList))

    def removeFromGraph(self, userList):
        self.graph.remove_nodes_from(userList)

    def reindexUsersInEventLog(self, dictionary):
        self.event_log.user = self.event_log.user.map(dictionary)

    def reindexContagionID(self, dictionary):
        self.event_log = self.event_log.assign(contagionID=self.event_log['contagion'].map(dictionary))

    def updateEventLog(self):
        if 'contagionID' in self.event_log.columns:
            t = defaultdict(functools.partial(next, itertools.count()))
            self.reindexContagionID(t)
            self.num_contagions = max(t.values()) + 1
            self.contagion_id_dict = t
        else:
            self.num_contagions = len(self.event_log.contagion.unique())
        self.num_events = self.event_log.shape[0]

    def reindexUsersInGraph(self, dictionary):
        self.graph = nx.relabel_nodes(self.graph, dictionary, copy=True)

    def reindexUsers(self):
        t = defaultdict(functools.partial(next, itertools.count()))
        self.edges.user1 = self.edges.user1.map(t)
        self.edges.user2 = self.edges.user2.map(t)
        self.reindexUsersInEventLog(t)
        if self.graph is not None:
            self.reindexUsersInGraph(t)
        self.num_users = max(t.values()) + 1

    def prepareTestData(self, fraction):
        minTime = self.event_log.ts.min()
        maxTime = self.event_log.ts.max()
        stoppingTime = float(maxTime - minTime) * fraction + minTime
        tmp = self.event_log[self.event_log.ts > stoppingTime]
        self.event_log = self.event_log[self.event_log.ts <= stoppingTime]
        self.num_events = self.event_log.shape[0]
        return tmp

    @staticmethod
    def fromPickle(directory):
        return pickle.load(open(directory + 'data.p', 'rb'))
