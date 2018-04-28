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

    def fit(self, data: Data, **kwargs):
        self._estimate(data, **kwargs)

    def predict(self, nam_iterations):
        self._predict(nam_iterations)

    @abstractmethod
    def _predict(self,num_iterations):
        pass

    @abstractmethod
    def _estimate(self, data, **kwargs):
        pass


class MultiContagionDynamicThresholdModel(BaseMultiContagionDiffusionModel):
    """
    The base class for Mutli-Contagion Diffusion of Information MultiContagionDynamicThresholdModel.

    A MultiContagionDynamicThresholdModel stores all the model parameters required to perform prediction of multi-contagious diffusion precess.

    Attributes
    ----------

    Methods
    -------


    """

    def __init__(self):

        self.contagion_correlation = ContagionCorrelation()
        self.adjacency_matrix = Adjacency()
        self.thresholds_matrix = Threshold()

    def _estimate(self, data, **kwargs):
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

    def toPickle(self, directory):
        pickle.dump(self, open(directory + 'MultiContagionDynamicThresholdModel.p', 'wb'))

    @staticmethod
    def from_pickle(directory):
        return pickle.load(open(directory+'MultiContagionDynamicThresholdModel.p','rb'))

    def _predict(self, num_iterations):
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
        for i in np.unique(np.where(activation_candidates[:, :] == True)[0]):  # iteracja po użytkownikach, którzy mają przekroczony próg
            temp1 = np.where(activation_candidates[i, :] == True)[0]  # tagi, w których dla użytkownika i przekroczony był próg
            temp2 = np.where(self.state_matrix_.matrix[i][:] == True)[0]  # tagi juz aktywne
            temp1 = np.setdiff1d(temp1, temp2)  # usuniecie juz aktywnych tagow
            if (not np.any(self.contagion_correlation.matrix[temp1[:, None], temp1] < 0)) and (
            not temp1.size == 0):  # sprawdzenie, czy kandydaci do aktywacji nie są negatywnie skorelowani
                # print('YES! ',l)
                self.state_matrix_.matrix[i][temp1] = True  # aktywacja uzytkownika i w tagach z listy temp1
                self.activity_index_vector_[i] += 1  # Y[i]+=1 #zwiekszenie licznika aktywacji uzytkownika i
                num_activations += 1
                for contagion in range(self.state_matrix_.num_contagions):  # temporary solution
                    self.thresholds_matrix.matrix[i][contagion] = 1 - math.pow(
                        1 - self.thresholds_matrix.initial_matrix[i][contagion],
                        self.activity_index_vector_[i] + 1)  # aktualizacja thety

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
