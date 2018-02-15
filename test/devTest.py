import model.parameters as model


def main():
    cm = model.CorrelationMatrix(100)
    cm.generateRandomCorrelationMatrix()
    print(cm.testSymetry())

if __name__ == "__main__":
    main()



