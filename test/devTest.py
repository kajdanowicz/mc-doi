import model.parameters as model
import data.Events as events
import config.config as config


directory = config.local['directory']

def main():
    # testLoadData(eventsFile)
    # testEliminationOfEventsWithMinOccurTags(fileName=eventsFile, minOccur=10)
    testEstimateCorrelationMatrixFromData(fileName=directory+'eventLog', minOccur=40000)

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