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

            .. _PEP 484:
                https://www.python.org/dev/peps/pep-0484/

            """
        C = np.random.random((self.size, self.size))
        C = C * 2 - 1
        C *= np.tri(*C.shape, k=-1)
        C = C + np.transpose(C) + self.correlationMatrix
        self.correlationMatrix = C

