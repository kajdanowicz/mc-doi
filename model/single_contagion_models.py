from abc import abstractmethod
from model.multi_contagion_models import *
from model.parameters import Adjacency
from model.parameters import ThresholdSingleContagion
from model.results import SingleIterResult
from model.results import Results
from data.data import Data
import numpy as np


class BaseSingleContagionDiffusionModel:
    """
    Base class for multi-contagion diffusion models. Each class inheriting from
    :class:`BaseSingleContagionDiffusionModel` should have :name:`fit` method and :name:`predict` method implemented.
    """

    @abstractmethod
    def fit(self, data, **kwargs):
        """
        Base method for fitting model's parameters. It evaluates model's specific methods.
        """
        pass

    @abstractmethod
    def predict(self, num_iterations: int):
        """
        Base method for prediction of information diffusion in single-contagion world. It evaluates model's specific
        prediction methods.
        """
        pass


class SingleContagionDynamicThresholdModel(MultiContagionDynamicThresholdModel):

    def __init__(self):

        super(SingleContagionDynamicThresholdModel, self).__init__()
        # self.adjacency = Adjacency()
        self.thresholds = ThresholdSingleContagion()
        self.contagion_correlation = None

    def fit(self, data: Data, **kwargs):
        """
        Fit Single-Contagion Dynamic Threshold models parameters according to :name:`data`. Method evaluates parameters specific
        estimation procedures.

        Parameters
        ----------
        data : Data
            :class:`Data` object according to which parameters should be fitted.
        **kwargs
            Arbitrary keyword arguments.
        """
        if (self.adjacency.matrix is None) and (self.thresholds.matrix is None):
            self.estimate_adjacency_matrix(data)
            # print('Adjacency')
            self.estimate_threshold_matrix(data, adjacency = self.adjacency, **kwargs)
            # print('Threshold')
            self.fill_state_matrix(data)
            # print('State')
        else:
            raise NameError('Can not estimate parameters when any of them is already assigned')

    def fill_state_matrix(self, data):
        # TODO state_matrix_.matrix -> sparse
        self.state_matrix_ = StateMatrix()
        self.state_matrix_.num_contagions = data.num_contagions
        self.state_matrix_.num_users = data.num_users
        self.state_matrix_.matrix = np.full((self.state_matrix_.num_users, self.state_matrix_.num_contagions), False, dtype=bool)
        for index, row in data.event_log.iterrows():
            self.state_matrix_.matrix[row[Data.user]][row[Data.contagion_id]] = True
        self.activity_index_vector_ = np.sum(self.state_matrix_.matrix, axis=1)

    def estimate_threshold_matrix(self, data: Data, adjacency, **kwargs):
        self.thresholds.estimate(data, adjacency=adjacency, **kwargs)

    def fit_only_thresholds_states(self, data: Data, **kwargs):
        if self.adjacency.matrix is not None:
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
            # print('Threshold')
            self.fill_state_matrix(data)
            # print('State')
        else:
            raise NameError('Can not estimate threshold - contagion correlation matrix or adjacency matrix not assigned')

    def __activation_matrix(self, influence_matrix):
        return influence_matrix

    def __check_negative_contagion_correlation(self, contagions_above_threshold_not_active):
        return contagions_above_threshold_not_active.size > 0
