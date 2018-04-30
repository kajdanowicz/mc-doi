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

class Adjacency(BaseParameter):
    def __init__(self):
        super(Adjacency, self).__init__()
        self.matrix = None
        self.matrix_transposed = None
        self.num_users = None
        self.event_queue = dict()
        self.v_2_u = None
        self.v_and_u = None
        self.u = None
