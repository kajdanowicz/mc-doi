import model.parameters.CorrelationMatrix


def main():
    cm = CorrelationMatrix(100)
    cm.generateRandomCorrelationMatrix()
    print(cm.correlationMatrix)

if __name__ == "__main__":
    main()



