import pandas as pd

class EventsData():

    def __init__(self, fileName=None, dataFrame=None):
        if dataFrame is not None:
            self.data = dataFrame
        elif fileName is not None:
            self.data = self.loadCSV(fileName)
        else:
            self.data = pd.DataFrame()

    def loadCSV(self, fileName):
        return EventsData(pd.read_csv(fileName))

    def getEventsMinOccurences(self, minOccur):
        ''' Return events that uses tags appearing in the data minOccur times.'''
        temp = self.data.groupby(by='tag').count().reset_index()[['tag', 'ts']]
        series = temp[(temp['ts'] > minOccur)]['tag']
        temp = self.data[self.data['tag'].isin(series)]
        return (temp)