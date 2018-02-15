import model.parameters as model
import data.Events as events


eventsFile = '/home/tomek/Dokumenty/Documents/Nauka/Dane/mc-doi_data/df_tag'

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