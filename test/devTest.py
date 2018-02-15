import model.parameters as model
import data.Events as events
import config.config as config


eventsFile = config.local['eventsFile']

def main():
    # testLoadData(eventsFile)
    testEliminationOfEventsWithMinOccurTags(fileName=eventsFile, minOccur=10)

def testCreationOfRandomCorrelationMatrix(size=100):
    cm = model.CorrelationMatrix(size)
    print(cm.generateRandomCorrelationMatrix())

def testSymetry(cm):
    print(cm.testSymetry())

def testLoadData(fileName):
    frame = events.EventsData(fileName=fileName)
    print(frame.data.shape)


def testEliminationOfEventsWithMinOccurTags(fileName, minOccur = 10):
    frame = events.EventsData(fileName=fileName)
    print(frame.getEventsMinOccurences(minOccur).shape)



if __name__ == "__main__":
    main()