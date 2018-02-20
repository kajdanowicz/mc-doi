import pandas as pd
import numpy as np
from collections import defaultdict


class data():

    def __init__(self):
        self.eventLog=None
        self.edges=None
        self.numUsers=None
        self.numContagions=None
        self.numEvents=None

    # def loadEventLogFile(self, filePath):
    #     self.eventLog=pd.read_csv(filePath)
    #     self.eventLog.columns=['ts','user','contagion']
    #     self.numUsers=len(self.eventLog['user'].unique()
    #     self.numContagions=len(self.eventLog['contagion'].unique())
    #     self.numEvents=self.eventLog.shape[0]
    #     # TODO Implement this method
    #     # TODO Column names
    #
    # def loadEventLogDataFrame(self, dataFrame):
    #     self.eventLog=dataFrame
    #     self.eventLog.columns=['ts','user','contagion']
    #     self.numUsers=len(self.eventLog['user'].unique()
    #     self.numContagions=len(self.eventLog['contagion'].unique())
    #     self.numEvents=self.eventLog.shape[0]
    #     # TODO Implement this method
    #     # TODO Column names

    # def loadEdgesFile(self, filePath):
    #     self.edges=pd.read_csv(filePath)
    #     # TODO Implement this method
    #
    # def loadEdgesDataFrame(self, dataFrame):
    #     # TODO Implement this method

    def loadDataFile(self,directory):
        eventLog=pd.read_csv(directory+'eventLog')
        eventLog.columns = ['ts', 'user', 'contagion']
        edges=pd.read_csv(directory+'edges')
        edges.columns = ['user1','user2']
        if verifyUsersCorrect(eventLog,edges)==True:
            self.eventLog=eventLog
            self.edges=edges
            self.numUsers=len(self.eventLog['user'].unique()) # fixme
            self.numContagions=len(self.eventLog['contagion'].unique())
            self.numEvents=self.eventLog.shape[0]
            return True
        else:
            return False
        # TODO Implement Exception
        # review

    def loadDataDataFrame(self, eventLogDF, edgesDF):
        eventLog=eventLogDF
        eventLog.columns = ['ts', 'user', 'contagion']
        edges=edgesDF
        edges.columns = ['user1','user2']
        if verifyUsersCorrect(eventLog,edges)==True:
            self.eventLog=eventLog
            self.edges=edges
            self.numUsers=len(self.eventLog['user'].unique()) # fixme
            self.numContagions=len(self.eventLog['contagion'].unique())
            self.numEvents=self.eventLog.shape[0]
            return True
        else:
            return False
        # TODO Implement Exception
        # review

    def verifyUsersCorrect(eventLogDF,edgesDF):
        if np.setdiff1d(eventLogDF['user'],np.union1d([edgesDF['user1'],edgesDF['user2']])).shape[0]==0:
            return True
        else
            return False

    # def verifyDataCorrect(self):
    #     # TODO Implement this method

    def loadData(self, directory=None, eventLogDF=None, edgesDF=None):
        ''' Loads data to class data instance from the source that depends on given arguments'''
        if directory is not None:
            if self.loadDataFile(directory):
                return True
            else:
                return False
        elif (eventLogDF is not None) and (edgesDF is not None):
            if self.loadDataDataFrame(eventLogDF,edgesDF):
                return True
            else:
                return False
        else:
            return False
        # review

    def loadDataMinOccurrence(self, minOccurs, directory=None, eventLogDF=None, edgesDF=None):
        """ Loads data to class data instance from the source that depends on given arguments
        Only contagions appearing in minOccurs events are loaded"""
        if self.loadData(directory,eventLogDF,edgesDF) == False:
            return False
        self.restrictEventLogMinOccurences(minOccurs)
        return True
        # review

    def restrictEventLogMinOccurences(self, minOccurs):
        """ Restricts events in self to that, which contains contagions appearing in the data minOccurs times."""
        temp = self.eventLog.groupby(by='contagion').count().reset_index()[['contagion', 'ts']]
        series = temp[(temp['ts'] > minOccurs)]['contagion']
        self.eventLog = self.eventLog[self.eventLog['contagion'].isin(series)]
        self.numContagions=len(series)
        self.numEvents=self.eventLog.shape[0]
        # t = defaultdict(lambda: len(t))
        # temp['tagID'] = temp.apply(lambda row: t[row['contagion']], axis=1)
        # review

    def deleteUsers(self,userList):
        self.edges.drop(self.edges[(self.edges['user1'].isin(userList)) | (self.edges['user2'].isin(userList))].index,inplace=True)
        self.eventLog.drop(self.eventLog[self.eventLog['user'].isin(userList)].index,inplace=True)
        self.numUsers=len(np.union1d(self.edges['user1'],self.edges['user2']))
        self.numEvents=self.eventLog.shape[0]
        self.numContagions=len(self.eventLog['contagion'].unique())
        # review

    def dropEdge(self,edge):
        self.edges.drop(self.edges[((self.edges['user1']==edge[0]) & (self.edges['user2']==edge[1])) | ((self.edges['user1']==edge[1]) & (self.edges['user2']==edge[0]))].index,inplace=True)

    def deleteEdges(self,edgesList):
        # question Should an unconnected user be deleted?
        for edge in edgesList:
            self.dropEdge(edge)
        # review

    def deleteContagions(self,contagionList):
        self.eventLog.drop(self.eventLog[self.eventLog['contagion'].isin(contagionList)].index,inplace=True)
        self.numEvents=self.eventLog.shape[0]
        self.numContagions=len(self.eventLog['contagion'].unique())
        # review
