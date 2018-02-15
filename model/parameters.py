import numpy as np

class CorrelationMatrix():
    def __init__(self, size):
        self.size = size
        self.correlationMatrix = np.eye(N=size)
        print('Init')

    def generateRandomCorrelationMatrix(self):
        """Function to generate random correlation matrix.

            Args:
                param1 (int): The first parameter.
                param2 (str): The second parameter.

            Returns:
                bool: The return value. True for success, False otherwise.
            """
        C = np.random.random((self.size, self.size))
        C = C * 2 - 1
        C *= np.tri(*C.shape, k=-1)
        C = C + np.transpose(C) + self.correlationMatrix
        self.correlationMatrix = C

    def testSymetry(self):
        for i in range(self.size):
            for j in range(i+1,self.size):
                if self.correlationMatrix[i][j]!=self.correlationMatrix[j][i]:
                    return False

        return True