import numpy as np
import math
import pickle
from model.cc_matrix import cc_matrix
from model.a_matrix import a_matrix
from model.t_matrix import t_matrix
from model.single_iter_result import single_iter_result
from model.results import results


class model():

    def __init__(self):
        '''

        '''
        self.contagion_correlation_matrix = cc_matrix()
        self.adjacency_matrix = a_matrix()
        self.thresholds_matrix = t_matrix()
        self.state_matrix = None  # macierz indykatorow
        self.activity_index_vector = None  # wykladnik

    def fit(self, data, batch_type, batch_size):
        # TODO Implement this method
        if self.contagion_correlation_matrix.matrix is None:
            self.estimate_contagion_correlation_matrix(data)
        if self.adjacency_matrix.matrix is None:
            self.estimate_adjacency_matrix(data)
        if batch_type == 'time':
            self.thresholds_matrix.estimate_time_batch(data, self.adjacency_matrix, self.contagion_correlation_matrix,
                                                       batch_size)
        elif batch_type == 'volume':
            self.thresholds_matrix.estimate_volume_batch(data, self.adjacency_matrix, self.contagion_correlation_matrix,
                                                         batch_size)
        elif batch_type == 'hybrid':
            self.thresholds_matrix.estimate_hybride_batch(data)
        self.fill_state_matrix(data)

    def fill_state_matrix(self, data):
        self.state_matrix = single_iter_result()
        self.state_matrix.num_contagions = data.num_contagions
        self.state_matrix.num_users = data.num_users
        self.state_matrix.matrix = np.full((self.state_matrix.num_users, self.state_matrix.num_contagions), False, dtype=bool)
        for index, row in data.event_log.iterrows():
            self.state_matrix.matrix[row['user']][row['contagion_id']] = True
        self.activity_index_vector = np.sum(self.state_matrix.matrix, axis=1)

    def estimate_contagion_correlation_matrix(self, data):
        self.contagion_correlation_matrix.estimate(data)

    def estimate_adjacency_matrix(self, data):
        self.adjacency_matrix.estimate(data)

    def toPickle(self, directory):
        pickle.dump(self, open(directory + 'model.p', 'wb'))

    @staticmethod
    def from_pickle(directory):
        return pickle.load(open(directory+'model.p','rb'))

    def predict(self, num_iterations):
        num_activations = 0
        r = results()
        self.adjacency_matrix.transpose()
        for l in range(num_iterations):
            U = self.adjacency_matrix.matrix_transposed.dot(self.state_matrix.matrix)
            F = U.dot(self.contagion_correlation_matrix.matrix) / self.contagion_correlation_matrix.num_contagions
            temp = np.greater_equal(F, self.thresholds_matrix.matrix)  # porównanie funkcji aktywacji z progiem
            ### dodawanie rekordów bez przekroczenia progu
            for i in np.unique(np.where(temp[:, :] == True)[0]):  # iteracja po użytkownikach, którzy mają przekroczony próg
                temp1 = np.where(temp[i, :] == True)[0]  # tagi, w których dla użytkownika i przekroczony był próg
                temp2 = np.where(self.state_matrix.matrix[i][:] == True)[0]  # tagi juz aktywne
                temp1 = np.setdiff1d(temp1, temp2)  # usuniecie juz aktywnych tagow
                if (not np.any(self.contagion_correlation_matrix.matrix[temp1[:, None], temp1] < 0)) and (not temp1.size == 0):  # sprawdzenie, czy kandydaci do aktywacji nie są negatywnie skorelowani
                    # print('YES! ',l)
                    self.state_matrix.matrix[i][temp1] = True  # aktywacja uzytkownika i w tagach z listy temp1
                    self.activity_index_vector[i] += 1  # Y[i]+=1 #zwiekszenie licznika aktywacji uzytkownika i
                    num_activations += 1
                    for contagion in range(self.state_matrix.num_contagions): #temporary solution
                        self.thresholds_matrix.matrix[i][contagion] = 1 - math.pow(1 - self.thresholds_matrix.initial_matrix[i][contagion], self.activity_index_vector[i] + 1)  # aktualizacja thety
            r.add_result(self.state_matrix)
        print(num_activations)
        return r

    def assign_contagions_correlation_matrix(self, matrix):
        # TODO Implement this method
        if self.state_matrix is None:
            self.contagion_correlation_matrix = matrix
        else:
            if self.state_matrix.shape[1] == matrix.shape[1]:
                self.contagion_correlation_matrix = matrix

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

    def model_iteration(self):
        # TODO Implement this method
        pass
