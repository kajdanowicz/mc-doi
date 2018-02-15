import numpy as np

class CorrelationMatrix():
    def __init__(self, size):
        self.size = size
        self.correlationMatrix = np.eye(N=size)
        print('Init')

    def generateRandomCorrelationMatrix(self):
        """Example function with types documented in the docstring.

            `PEP 484`_ type annotations are supported. If attribute, parameter, and
            return types are annotated according to `PEP 484`_, they do not need to be
            included in the docstring:

            Args:
                param1 (int): The first parameter.
                param2 (str): The second parameter.

            Returns:
                bool: The return value. True for success, False otherwise.

            .. _PEP 484:
                https://www.python.org/dev/peps/pep-0484/

            """
        self.correlationMatrix = Maciek

