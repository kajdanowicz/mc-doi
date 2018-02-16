import pandas as pd
from collections import defaultdict


class data():

    def __init__(self):
        self.eventLog=None
        self.edges=None
        self.numUsers=None
        self.numContagions=None
        self.numEvents=None

    def loadEventLogFile(self, filePath):
        # TODO Implement this method
        # TODO Column names

    def loadEventLogDataFrame(self, dataFrame):
        # TODO Implement this method
        # TODO Column names

    def loadEdgesFile(self, filePath):
        # TODO Implement this method

    def loadEdgesDataFrame(self, dataFrame):
        # TODO Implement this method

    def loadDataFile(self,directory):
        # TODO Implement this method

    def loadDataDataFrame(self, dfEventLog, dfEdges):
        # TODO Implement this method

    def verifyDataCorrect(self):
        # TODO Implement this method

    def loadData(self, minOccur):
        # TODO Implement this method
        ''' Loads data to class data instance from the source that depends on given arguments
        Only contagions appearing in minOccur events are loaded'''

    def restrictEventLogMinOccurences(self, minOccur):
        ''' Return events that uses contagions appearing in the data minOccur times.'''
        # TODO Column names
        temp = self.eventLog.groupby(by='tag').count().reset_index()[['tag', 'ts']]
        series = temp[(temp['ts'] > minOccur)]['tag']
        temp = self.eventLog[self.eventLog['tag'].isin(series)]
        t = defaultdict(lambda: len(t))
        temp['tagID'] = temp.apply(lambda row: t[row['tag']], axis=1)
        u = defaultdict(lambda: len(u))
        temp.apply(lambda row: u[row['user']], axis=1)
        temp['user'] = temp['user'].map(u)
        # review