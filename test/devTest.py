import model.parameters as model
import data.Events as events

def main():
    testLoadData()





def testCreationOfRandomCorrelationMatrix(size=100):
    cm = model.CorrelationMatrix(size)
    print(cm.generateRandomCorrelationMatrix())

def testSymetry(cm):
    print(cm.testSymetry())

def testLoadData(fileName):
    print(len(events(fileName)))



if __name__ == "__main__":
    main()