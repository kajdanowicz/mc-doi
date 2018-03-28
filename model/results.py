import copy

class results():

    def __init__(self):
        self.list = []
        self.numResults = 0

    def addResult(self,result):
        self.list.append(copy.copy(result))
        self.numResults += 1

    def getResult(self,resultNum):
        if resultNum >= self.numResults:
            return False
        else:
            return self.list[resultNum]