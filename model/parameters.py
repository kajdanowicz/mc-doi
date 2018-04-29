from abc import abstractmethod
from data.data import Data
from numpy import ndarray


class BaseParameter:
    """
    Base class for model's parameters
    """
    def __init__(self):
        self.matrix = None

    @abstractmethod
    def estimate(self, data: Data):
        pass

    @abstractmethod
    def assign_matrix(self,matrix: ndarray):
        pass
