import model.parameters as model
import data.Events as events

def main():
    testLoadData('/home/tomek/Dokumenty/Documents/Nauka/Dane/mc-doi_data/df_tag')

def testCreationOfRandomCorrelationMatrix(size=100):
    cm = model.CorrelationMatrix(size)
    print(cm.generateRandomCorrelationMatrix())

def testSymetry(cm):
    print(cm.testSymetry())

def testLoadData(fileName):
    frame = events.EventsData(fileName=fileName)
    print(frame.data.shape)





if __name__ == "__main__":
    main()