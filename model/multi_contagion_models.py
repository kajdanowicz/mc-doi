import numpy as np
import math
import pickle
from model.parameters import ContagionCorrelation
from model.parameters import Adjacency
from model.parameters import Threshold
from model.results import SingleIterResult
from model.results import Results
from data.data import Data
from abc import abstractmethod


class BaseMultiContagionDiffusionModel:
    """
    Base class for multi-contagion diffusion models. Each class inheriting from
    :class:`BaseMultiContagionDiffusionModel` should have :name:`fit` and :name:`predict` methods.
    """

    @abstractmethod
    def fit(self, data: Data, **kwargs):
        """
        Base method for fitting model's parameters. It evaluates model's specific methods.
        """
        pass

    @abstractmethod
    def predict(self, num_iterations: int) -> Results:
        """
        Base method for prediction of information diffusion in multi-contagion world. It evaluates model's specific
        prediction methods.
        """
        pass


class MultiContagionDynamicThresholdModel(BaseMultiContagionDiffusionModel):
    """
    The base class for Mutli-Contagion Diffusion of Information MultiContagionDynamicThresholdModel.

    A MultiContagionDynamicThresholdModel stores all the model parameters required to perform prediction of
    multi-contagious diffusion precess.

    Attributes
    ----------
    contagion_correlation : ContagionCorrelation
        Stores the contagion correlation matrix of contagions in event log.
    adjacency : Adjacency
        Stores the adjacency matrix of the underlying social network.
    thresholds : Threshold
        Stores dynamic threshold of all users in the form of a matrix. Entries for specific user are equal
        across all columns.
    state_matrix_ : SingleIterResult
        Stores the current state of the network in the sense of users activity in particular contagions.
    activity_index_vector_ : numpy.array
        Stores the current number of activations performed by each user.
    """

    def __init__(self):

        self.contagion_correlation = ContagionCorrelation()
        self.adjacency = Adjacency()
        self.thresholds = Threshold()

    def fit(self, data: Data, **kwargs):
        """
        Fit Multi-Contagion Dynamic Threshold models parameters according to :name:`data`. Method evaluates parameters specific
        estimation procedures.

        Parameters
        ----------
        data : Data
            :class:`Data` object according to which parameters should be fitted.
        **kwargs
            Arbitrary keyword arguments.
        """
        if (self.contagion_correlation.matrix is None) and (self.adjacency.matrix is None) and (self.thresholds.matrix is None):
            self.estimate_contagion_correlation_matrix(data)
            # print('ContagionCorrelation')
            self.estimate_adjacency_matrix(data)
            # print('Adjacency')
            self.estimate_threshold_matrix(data, adjacency = self.adjacency, correlation = self.contagion_correlation, **kwargs)
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

    def estimate_contagion_correlation_matrix(self, data):
        self.contagion_correlation.estimate(data)

    def estimate_adjacency_matrix(self, data: Data, **kwargs):
        self.adjacency.estimate(data, **kwargs)

    def estimate_threshold_matrix(self, data: Data, adjacency, correlation, **kwargs):
        self.thresholds.estimate(data, adjacency=adjacency, correlation=correlation, **kwargs)

    def to_pickle(self, directory):
        # TODO directory + ... -> fileName
        pickle.dump(self, open(directory + 'MultiContagionDynamicThresholdModel.p', 'wb'))

    @staticmethod
    def from_pickle(directory):
        # TODO directory + ... -> fileName
        return pickle.load(open(directory+'MultiContagionDynamicThresholdModel.p','rb'))

    def predict(self, num_iterations: int) -> Results:
        # TODO "method" rst
        # TODO replace num_activation by proper test
        """

        Parameters
        ----------
        num_iterations : int
            Discrete number of prediction iteration steps to perform by :method:predict method

        Returns
        -------
        Results
            Object containing results from all predication iterations.

        """
        global num_activations
        num_activations = 0
        result = Results()
        self.adjacency.transposed()
        for l in range(num_iterations):
            result.add_result(self.__single_iteration())
        # print(num_activations)
        return result

    def __single_iteration(self) -> SingleIterResult:
        # TODO SingleIterResult -> new special class
        influence_matrix = self.__influence_matrix()
        activation_matrix = self.__activation_matrix(influence_matrix)
        self.__activation_procedure(activation_matrix)
        return self.state_matrix_

    def __activation_procedure(self, activation_matrix):
        # TODO delete num_activations
        global num_activations
        activation_candidates = self.__find_activation_candidates(activation_matrix)
        for user in self.__users_above_threshold(activation_candidates):
            contagions_above_threshold = self.__contagions_above_threshold(activation_candidates,
                                                      user)  # contagions for user in which threshold has been exceeded
            active_contagions = self.active_contagions(user)  # contagions in which user is already active
            contagions_above_threshold_not_active = self.__contagions_above_threshold_not_active(active_contagions,
                                                                                                 contagions_above_threshold)  # delete active_contagions from contagions_above_threshold
            if self.__check_negative_contagion_correlation(contagions_above_threshold_not_active):  # check weather
                # candidates are not negatively correlated
                self.__activation(contagions_above_threshold_not_active, user)
                self.__increase_activity_index(user)
                num_activations += 1
                self.__update_threshold(user)

    def __check_negative_contagion_correlation(self, contagions_above_threshold_not_active):
        # TODO review of correctness of condition
        return (not np.any(self.contagion_correlation.matrix[contagions_above_threshold_not_active[:,
                                                             None], contagions_above_threshold_not_active] < 0)) and (
                   not contagions_above_threshold_not_active.size == 0)

    def __update_threshold(self, user):
        # TODO assign vector in one line
        for contagion in range(self.state_matrix_.num_contagions):  # temporary solution
            self.thresholds.matrix[user][contagion] = 1 - math.pow(
                1 - self.thresholds.initial_matrix[user][contagion],
                self.activity_index_vector_[user] + 1)  # aktualizacja thety

    def __increase_activity_index(self, user):
        self.activity_index_vector_[user] += 1  # Y[user]+=1 #zwiekszenie licznika aktywacji uzytkownika user

    def __activation(self, contagions_above_threshold_not_active, user):
        self.state_matrix_.matrix[user][
            contagions_above_threshold_not_active] = True  # aktywacja uzytkownika user w tagach z listy contagions_above_threshold

    def __contagions_above_threshold_not_active(self, active_contagions, contagions_above_threshold):
        return np.setdiff1d(contagions_above_threshold, active_contagions)

    def active_contagions(self, user):
        return np.where(self.state_matrix_.matrix[user][:])[0]

    def __contagions_above_threshold(self, activation_candidates, user):
        return np.where(activation_candidates[user, :])[0]

    def __users_above_threshold(self, activation_candidates):
        return np.unique(np.where(activation_candidates[:, :])[0])

    def __find_activation_candidates(self, activation_matrix):
        return np.greater_equal(activation_matrix, self.thresholds.matrix)

    def __activation_matrix(self, influence_matrix):
        return influence_matrix.dot(self.contagion_correlation.matrix) / self.contagion_correlation.num_contagions_

    def __influence_matrix(self):
        return self.adjacency.matrix_transposed_.dot(self.state_matrix_.matrix)

    def assign_contagions_correlation_matrix(self, matrix):
        self.contagion_correlation.assign_matrix(matrix)

    def assign_adjacency_matrix(self, adjacency_matrix):
        # TODO Implement this method
        pass

    def assign_thresholds_matrix(self, thresholds_vector):
        # TODO Implement this method
        pass

    def assign_state_matrix(self, state_matrix):
        # TODO Implement this method
        pass

    def assign_activity_index_vector(self, activity_index_vector):
        # TODO Implement this method
        pass


class StateMatrix(SingleIterResult):
    # TODO Write docstring
    def __init__(self):
        super(StateMatrix, self).__init__()