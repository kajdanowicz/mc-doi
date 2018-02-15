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
        return EventsData(dataFrame=pd.read_csv(fileName))

    def getEventsMinOccurences(self, minOccur):
        ''' Return events that uses tags appearing in the data minOccur times.'''



        return ()