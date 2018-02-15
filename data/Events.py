import pandas as pd

class EventsData():

    def __init__(self, fileName=None, dataFrame=None):
        self.data = pd.DataFrame()

        if dataFrame is not None:
            self.data = dataFrame
        elif fileName is not None:
            self.data = pd.read_csv(fileName)

    def loadCSV(self, fileName):
        return EventsData(dataFrame=pd.read_csv(fileName))


    def getEventsMinOccurences(self, minOccur):
        ''' Return events that uses tags appearing in the data minOccur times.'''
        temp = self.data.groupby(by='tag').count().reset_index()[['tag', 'ts']]
        series = temp[(temp['ts'] > minOccur)]['tag']
        temp = self.data[self.data['tag'].isin(series)]
        return (temp)