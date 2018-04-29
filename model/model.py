import numpy as np
import math
import pickle
from model.contagion_correlation import ContagionCorrelation
from model.adjacency import Adjacency
from model.threshold import Threshold
from model.results import SingleIterResult
from model.results import Results
from data.data import Data
from abc import abstractmethod


class BaseMultiContagionDiffusionModel:

    @abstractmethod
    def fit(self, data: Data, **kwargs):
        """
        Base method for fitting model's parameters. It evaluates model's specific methods.

        Parameters
        ----------
        data : Data
            Data object to which model's parameters are fitted. In contains an event log and the underlying social network.
        kwargs :
            Other parameters which specify types of algorithms used to while fitting.
        """
        pass

    @abstractmethod
    def predict(self, num_iterations: int) -> Results:
        """
        Base method for prediction of information diffusion in multi-contagion world. It evaluates model's specific
        prediction methods.

        Parameters
        ----------
        num_iterations :  int
            Number of iterations of prediction algorithm.
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
    adjacency_matrix : Adjacency
        Stores the adjacency matrix of the underlying social network.
    thresholds_matrix : Threshold
        Stores dynamic threshold of all users in the form of a matrix. Entries for specific user are equal
        across all columns.

    Methods
    -------


    """

    def __init__(self):

        self.contagion_correlation = ContagionCorrelation()
        self.adjacency_matrix = Adjacency()
        self.thresholds_matrix = Threshold()

    def fit(self, data, **kwargs):
        # TODO Implement this method
        if self.contagion_correlation.matrix is None:
            self.estimate_contagion_correlation_matrix(data)
            print('ContagionCorrelation')
        if self.adjacency_matrix.matrix is None:
            self.estimate_adjacency_matrix(data)
            print('Adjacency')
        if kwargs['batch_type'] == 'time':
            self.thresholds_matrix.estimate_time_batch(data, self.adjacency_matrix, self.contagion_correlation,
                                                       kwargs['batch_size'])
        elif kwargs['batch_type'] == 'volume':
            self.thresholds_matrix.estimate_volume_batch(data, self.adjacency_matrix, self.contagion_correlation,
                                                         kwargs['batch_size'])
        elif kwargs['batch_type'] == 'hybrid':
            self.thresholds_matrix.estimate_hybride_batch(data)
        print('Threshold')
        self.fill_state_matrix(data)
        print('State')

    def fill_state_matrix(self, data):
        self.state_matrix_ = SingleIterResult()
        self.state_matrix_.num_contagions = data.num_contagions
        self.state_matrix_.num_users = data.num_users
        self.state_matrix_.matrix = np.full((self.state_matrix_.num_users, self.state_matrix_.num_contagions), False, dtype=bool)
        for index, row in data.event_log.iterrows():
            self.state_matrix_.matrix[row['user']][row['contagion_id']] = True
        self.activity_index_vector_ = np.sum(self.state_matrix_.matrix, axis=1)

    def estimate_contagion_correlation_matrix(self, data):
        self.contagion_correlation.estimate(data)

    def estimate_adjacency_matrix(self, data):
        self.adjacency_matrix.estimate(data)

    def to_pickle(self, directory):
        pickle.dump(self, open(directory + 'MultiContagionDynamicThresholdModel.p', 'wb'))

    @staticmethod
    def from_pickle(directory):
        return pickle.load(open(directory+'MultiContagionDynamicThresholdModel.p','rb'))

    def predict(self, num_iterations: int) -> Results:
        global num_activations
        num_activations = 0
        r = Results()
        self.adjacency_matrix.transpose()
        for l in range(num_iterations):
            self._single_iteration(r)
        print(num_activations)
        return r

    def _single_iteration(self, r):
        influence_matrix = self._influence_matrix()
        activation_matrix = self._activation_matrix(influence_matrix)
        self._activation_procedure(activation_matrix)
        r.add_result(self.state_matrix_)

    def _activation_procedure(self, activation_matrix):
        global num_activations
        activation_candidates = self.__find_activation_candidates(activation_matrix)
        for user in self.__users_above_threshold(activation_candidates):  # iteracja po użytkownikach, którzy mają przekroczony próg
            contagions_above_threshold = self.__contagions_above_threshold(activation_candidates,
                                                      user)  # tagi, w których dla użytkownika user przekroczony był próg
            active_contagions = self.active_contagions(user)  # tagi juz aktywne
            contagions_above_threshold_not_active = self.__contagions_above_threshold_not_active(active_contagions,
                                                                                                 contagions_above_threshold)  # usuniecie juz aktywnych tagow
            if (not np.any(self.contagion_correlation.matrix[contagions_above_threshold_not_active[:, None], contagions_above_threshold_not_active] < 0)) and (
            not contagions_above_threshold_not_active.size == 0):  # sprawdzenie, czy kandydaci do aktywacji nie są negatywnie skorelowani
                self.__activation(contagions_above_threshold_not_active, user)
                self.__increase_activity_index(user)
                num_activations += 1
                self.__update_threshold(user)

    def __update_threshold(self, user):
        for contagion in range(self.state_matrix_.num_contagions):  # temporary solution
            self.thresholds_matrix.matrix[user][contagion] = 1 - math.pow(
                1 - self.thresholds_matrix.initial_matrix[user][contagion],
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
        return np.greater_equal(activation_matrix, self.thresholds_matrix.matrix)

    def _activation_matrix(self, influence_matrix):
        return influence_matrix.dot(self.contagion_correlation.matrix) / self.contagion_correlation.num_contagions

    def _influence_matrix(self):
        return self.adjacency_matrix.matrix_transposed.dot(self.state_matrix_.matrix)

    def assign_contagions_correlation_matrix(self, matrix):
        # TODO Implement this method
        pass

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
