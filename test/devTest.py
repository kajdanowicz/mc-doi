import model.parameters as model
import data.data as data
import config.config as config

directory = config.local['directory']

def main():
    testRestrictEventLogMinOccurences(directory,minOccurs=50000)


def test(t):
    if t:
        print('Passed')
    else:
        print('Failed')

def testDeleteUsers(userList):
    d=data.data()
    d.loadData(directory=directory)
    u=d.deleteUsers(userList)
    print(u.values())


def testLoadDataFile(directory):
    # passed
    d=data.data()
    return d.loadDataFile(directory)

def testLoadDataDataFrame(eventLogDF,edgesDF):
    d=data.data()
    return d.loadDataDataFrame(eventLogDF,edgesDF)

def testRestrictEventLogMinOccurences(directory, minOccurs = 40000):
    # TODO Design this test
    d=data.data()
    d.loadData(directory=directory)
    print('Before restriction:',d.eventLog.shape,'numContagions:',d.numContagions)
    d.restrictEventLogMinOccurences(minOccurs)
    print('After restriction:', d.eventLog.shape,'numContagions:',d.numContagions)


























def testCreationOfRandomCorrelationMatrix(size=100):
    cm = model.CorrelationMatrix()
    print(cm.generateRandomCorrelationMatrix(size))

def testSymetry(cm):
    print(cm.testSymetry())

def testLoadData(fileName):
    frame = events.EventsData(fileName=fileName)
    print(frame.data.shape)




def testEstimateCorrelationMatrixFromData(fileName, minOccur = 10):
    frame = events.EventsData(fileName=fileName)
    cm = model.CorrelationMatrix()
    cm.estimateCorrelationMatrixFromData(frame.getEventsMinOccurences(minOccur))
    print(cm.correlationMatrix)
    testSymetry(cm)

if __name__ == "__main__":
    main()