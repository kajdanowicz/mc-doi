import pandas as pd
import numpy as np
from collections import defaultdict


class data():

    def __init__(self):
        self.eventLog = None
        self.edges = None
        self.numUsers = None
        self.numContagions = None
        self.numEvents = None
        self.contagionIDDict=None

    # def loadEventLogFile(self, filePath):
    #     self.eventLog=pd.read_csv(filePath)
    #     self.eventLog.columns=['ts','user','contagion']
    #     self.numUsers=len(self.eventLog['user'].unique()
    #     self.numContagions=len(self.eventLog['contagion'].unique())
    #     self.numEvents=self.eventLog.shape[0]

    #
    # def loadEventLogDataFrame(self, dataFrame):
    #     self.eventLog=dataFrame
    #     self.eventLog.columns=['ts','user','contagion']
    #     self.numUsers=len(self.eventLog['user'].unique()
    #     self.numContagions=len(self.eventLog['contagion'].unique())
    #     self.numEvents=self.eventLog.shape[0]

    # def loadEdgesFile(self, filePath):
    #     self.edges=pd.read_csv(filePath)

    #
    # def loadEdgesDataFrame(self, dataFrame):

    def loadDataFile(self, directory):
        eventLogDF = pd.read_csv(directory + 'eventLog')
        eventLogDF.columns = ['ts', 'user', 'contagion']
        edgesDF = pd.read_csv(directory + 'edges')
        edgesDF.columns = ['user1', 'user2']
        if data.verifyUsersCorrect(eventLogDF, edgesDF):
            self.eventLog = eventLogDF
            self.edges = edgesDF
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

    def loadData(self, directory=None, eventLogDF=None, edgesDF=None):
        ''' Loads data to class data instance from the source that depends on given arguments'''
        if directory is not None:
            if self.loadDataFile(directory):
                return True
            else:
                return False
        elif (eventLogDF is not None) and (edgesDF is not None):
            if self.loadDataDataFrame(eventLogDF, edgesDF):
                return True
            else:
                return False
        else:
            return False
        # review

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


    def dropEdge(self, edge):
        self.edges.drop(self.edges[((self.edges['user1'] == edge[0]) & (self.edges['user2'] == edge[1])) | (
                    (self.edges['user1'] == edge[1]) & (self.edges['user2'] == edge[0]))].index, inplace=True)
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

    def addContagionID(self):
        t = defaultdict(lambda: len(t))
        self.eventLog['contagionID']=self.eventLog['contagion'].apply(lambda row: t[row['contagion']], axis=1)
        self.contagionIDDict=t
        # review

    def estimateContagionCorrelationMatrix(self):
        # TODO Implement
        # self.size = eventsLog['tagID'].max()+1
        # self.correlationMatrix = np.eye(N=self.size)
        numUsers=self.numUsers
        tmp = self.eventLog[['user', 'contagion']].drop_duplicates(subset=None, keep='first', inplace=False)
        allUsersIDs = np.linspace(0, numUsers, num=numUsers + 1).astype(int)
        for i in range(data.numContagions):
            for j in range(i + 1, data.numContagions):
                i_users = tmp[tmp['tagID'] == i]['user'].as_matrix()
                j_users = tmp[tmp['tagID'] == j]['user'].as_matrix()
                ni_users = np.setdiff1d(allUsersIDs, i_users)
                nj_users = np.setdiff1d(allUsersIDs, j_users)
                pij = len(np.intersect1d(i_users, j_users)) / numUsers
                pinj = len(np.intersect1d(i_users, nj_users)) / numUsers
                pnij = len(np.intersect1d(ni_users, j_users)) / numUsers
                pi = len(i_users) / numUsers
                pj = len(j_users) / numUsers
                pni = len(ni_users) / numUsers
                pnj = len(nj_users) / numUsers
                wynik = pij / math.sqrt(pi * pj) - (pnij / math.sqrt(pni * pj) + pinj / math.sqrt(pi * pnj)) / 2
                self.correlationMatrix[i][j] = wynik
                self.correlationMatrix[j][i] = wynik
