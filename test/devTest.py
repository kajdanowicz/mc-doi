import model.parameters as model
import data.data as data
import config.config as config

directory = config.local['directory']

def main():
    test(testLoadDataFile(directory))


def test(t):
    if t:
        print('Passed')
    else:
        print('Failed')

def testLoadDataFile(directory):
    # passed
    d=data.data()
    return d.loadDataFile(directory)

def testLoadDataDataFrame(eventLogDF,edgesDF):
    d=data.data()
    return d.loadDataDataFrame(eventLogDF,edgesDF)




























def testCreationOfRandomCorrelationMatrix(size=100):
    cm = model.CorrelationMatrix()
    print(cm.generateRandomCorrelationMatrix(size))

def testSymetry(cm):
    print(cm.testSymetry())

def testLoadData(fileName):
    frame = events.EventsData(fileName=fileName)
    print(frame.data.shape)


def testEliminationOfEventsWithMinOccurTags(fileName, minOccur = 10):
    frame = events.EventsData(fileName=fileName)
    print(frame.getEventsMinOccurences(minOccur).shape)

def testEstimateCorrelationMatrixFromData(fileName, minOccur = 10):
    frame = events.EventsData(fileName=fileName)
    cm = model.CorrelationMatrix()
    cm.estimateCorrelationMatrixFromData(frame.getEventsMinOccurences(minOccur))
    print(cm.correlationMatrix)
    testSymetry(cm)

if __name__ == "__main__":
    main()