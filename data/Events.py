import pandas as pd

class EventsData():

    def __init__(self, fileName):
        self.data = self.loadCSV(fileName)

    def loadCSV(self, fileName):
        return pd.read_csv(fileName)