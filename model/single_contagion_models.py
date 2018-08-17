from abc import abstractmethod
from model.parameters_DynamicLinearThreshold import Adjacency as Adjacency_DynamicLinearThreshold
from model.parameters_DynamicLinearThreshold import ThresholdSingleContagion as Threshold_DynamicLinearThreshold
from model.parameters_LinearThreshold import Adjacency as Adjacency_LinearThreshold
from model.parameters_LinearThreshold import ThresholdSingleContagion as Threshold_LinearThreshold
from model.parameters_LinearThreshold_random import Adjacency as Adjacency_LinearThreshold_random
from model.parameters_LinearThreshold_random import ThresholdSingleContagion as Threshold_LinearThreshold_random
from model.parameters_LinearThreshold_randomDynamic import Adjacency as Adjacency_LinearThreshold_randomDynamic
from model.parameters_LinearThreshold_randomDynamic import ThresholdSingleContagion as Threshold_LinearThreshold_randomDynamic
from model.parameters_LinearThreshold_random14all import Adjacency as Adjacency_LinearThreshold_random14all
from model.parameters_LinearThreshold_random14all import ThresholdSingleContagion as Threshold_LinearThreshold_random14all
from model.parameters_LinearThreshold_randomDynamic14all import Adjacency as Adjacency_LinearThreshold_randomDynamic14all
from model.parameters_LinearThreshold_randomDynamic14all import ThresholdSingleContagion as Threshold_LinearThreshold_randomDynamic14all
from model.multi_contagion_models import *
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


class SingleContagionDynamicLinearThresholdModel(MultiContagionDynamicLinearThresholdModel):

    def __init__(self):

        # super(SingleContagionDynamicLinearThresholdModel, self).__init__()
        self.adjacency = Adjacency_DynamicLinearThreshold()
        self.thresholds = Threshold_DynamicLinearThreshold()
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

    def __influence_matrix(self):
        return self.adjacency.matrix_transposed_.dot(self.state_matrix_.matrix)

    def __check_negative_contagion_correlation(self, contagions_above_threshold_not_active):
        return contagions_above_threshold_not_active.size > 0


class SingleContagionLinearThresholdModel(MultiContagionDynamicLinearThresholdModel):

        def __init__(self):

            # super(SingleContagionDynamicLinearThresholdModel, self).__init__()
            self.adjacency = Adjacency_LinearThreshold()
            self.thresholds = Threshold_LinearThreshold()
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
                self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
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
            self.state_matrix_.matrix = np.full((self.state_matrix_.num_users, self.state_matrix_.num_contagions),
                                                False, dtype=bool)
            for index, row in data.event_log.iterrows():
                self.state_matrix_.matrix[row[Data.user]][row[Data.contagion_id]] = True

        def estimate_threshold_matrix(self, data: Data, adjacency, **kwargs):
            self.thresholds.estimate(data, adjacency=adjacency, **kwargs)

        def fit_only_thresholds_states(self, data: Data, **kwargs):
            if self.adjacency.matrix is not None:
                self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
                # print('Threshold')
                self.fill_state_matrix(data)
                # print('State')
            else:
                raise NameError(
                    'Can not estimate threshold - contagion correlation matrix or adjacency matrix not assigned')

        def __activation_matrix(self, influence_matrix):
            return influence_matrix

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

        def __influence_matrix(self):
            return self.adjacency.matrix_transposed_.dot(self.state_matrix_.matrix)

        def __check_negative_contagion_correlation(self, contagions_above_threshold_not_active):
            return contagions_above_threshold_not_active.size > 0


class SingleContagionLinearThresholdModel_random(MultiContagionDynamicLinearThresholdModel):

    def __init__(self):

        # super(SingleContagionDynamicLinearThresholdModel, self).__init__()
        self.adjacency = Adjacency_LinearThreshold_random()
        self.thresholds = Threshold_LinearThreshold_random()
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
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
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
        self.state_matrix_.matrix = np.full((self.state_matrix_.num_users, self.state_matrix_.num_contagions),
                                            False, dtype=bool)
        for index, row in data.event_log.iterrows():
            self.state_matrix_.matrix[row[Data.user]][row[Data.contagion_id]] = True

    def estimate_threshold_matrix(self, data: Data, adjacency, **kwargs):
        self.thresholds.estimate(data, adjacency=adjacency, **kwargs)

    def fit_only_thresholds_states(self, data: Data, **kwargs):
        if self.adjacency.matrix is not None:
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
            # print('Threshold')
            self.fill_state_matrix(data)
            # print('State')
        else:
            raise NameError(
                'Can not estimate threshold - contagion correlation matrix or adjacency matrix not assigned')

    def __activation_matrix(self, influence_matrix):
        return influence_matrix

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

    def __influence_matrix(self):
        return self.adjacency.matrix_transposed_.dot(self.state_matrix_.matrix)

    def __check_negative_contagion_correlation(self, contagions_above_threshold_not_active):
        return contagions_above_threshold_not_active.size > 0


class SingleContagionLinearThresholdModel_randomDynamic(MultiContagionDynamicLinearThresholdModel):

    def __init__(self):

        # super(SingleContagionDynamicLinearThresholdModel, self).__init__()
        self.adjacency = Adjacency_LinearThreshold_randomDynamic()
        self.thresholds = Threshold_LinearThreshold_randomDynamic()
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
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
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
        self.state_matrix_.matrix = np.full((self.state_matrix_.num_users, self.state_matrix_.num_contagions),
                                            False, dtype=bool)
        for index, row in data.event_log.iterrows():
            self.state_matrix_.matrix[row[Data.user]][row[Data.contagion_id]] = True

    def estimate_threshold_matrix(self, data: Data, adjacency, **kwargs):
        self.thresholds.estimate(data, adjacency=adjacency, **kwargs)

    def fit_only_thresholds_states(self, data: Data, **kwargs):
        if self.adjacency.matrix is not None:
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
            # print('Threshold')
            self.fill_state_matrix(data)
            # print('State')
        else:
            raise NameError(
                'Can not estimate threshold - contagion correlation matrix or adjacency matrix not assigned')

    def __activation_matrix(self, influence_matrix):
        return influence_matrix

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
        self.thresholds.reestimate()

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

    def __influence_matrix(self):
        return self.adjacency.matrix_transposed_.dot(self.state_matrix_.matrix)

    def __check_negative_contagion_correlation(self, contagions_above_threshold_not_active):
        return contagions_above_threshold_not_active.size > 0


class SingleContagionLinearThresholdModel_random14all(MultiContagionDynamicLinearThresholdModel):

    def __init__(self):

        # super(SingleContagionDynamicLinearThresholdModel, self).__init__()
        self.adjacency = Adjacency_LinearThreshold_random14all()
        self.thresholds = Threshold_LinearThreshold_random14all()
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
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
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
        self.state_matrix_.matrix = np.full((self.state_matrix_.num_users, self.state_matrix_.num_contagions),
                                            False, dtype=bool)
        for index, row in data.event_log.iterrows():
            self.state_matrix_.matrix[row[Data.user]][row[Data.contagion_id]] = True

    def estimate_threshold_matrix(self, data: Data, adjacency, **kwargs):
        self.thresholds.estimate(data, adjacency=adjacency, **kwargs)

    def fit_only_thresholds_states(self, data: Data, **kwargs):
        if self.adjacency.matrix is not None:
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
            # print('Threshold')
            self.fill_state_matrix(data)
            # print('State')
        else:
            raise NameError(
                'Can not estimate threshold - contagion correlation matrix or adjacency matrix not assigned')

    def __activation_matrix(self, influence_matrix):
        return influence_matrix

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

    def __influence_matrix(self):
        return self.adjacency.matrix_transposed_.dot(self.state_matrix_.matrix)

    def __check_negative_contagion_correlation(self, contagions_above_threshold_not_active):
        return contagions_above_threshold_not_active.size > 0


class SingleContagionLinearThresholdModel_randomDynamic14all(MultiContagionDynamicLinearThresholdModel):

    def __init__(self):

        # super(SingleContagionDynamicLinearThresholdModel, self).__init__()
        self.adjacency = Adjacency_LinearThreshold_randomDynamic14all()
        self.thresholds = Threshold_LinearThreshold_randomDynamic14all()
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
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
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
        self.state_matrix_.matrix = np.full((self.state_matrix_.num_users, self.state_matrix_.num_contagions),
                                            False, dtype=bool)
        for index, row in data.event_log.iterrows():
            self.state_matrix_.matrix[row[Data.user]][row[Data.contagion_id]] = True

    def estimate_threshold_matrix(self, data: Data, adjacency, **kwargs):
        self.thresholds.estimate(data, adjacency=adjacency, **kwargs)

    def fit_only_thresholds_states(self, data: Data, **kwargs):
        if self.adjacency.matrix is not None:
            self.estimate_threshold_matrix(data, adjacency=self.adjacency, **kwargs)
            # print('Threshold')
            self.fill_state_matrix(data)
            # print('State')
        else:
            raise NameError(
                'Can not estimate threshold - contagion correlation matrix or adjacency matrix not assigned')

    def __activation_matrix(self, influence_matrix):
        return influence_matrix

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
        self.thresholds.reestimate()

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

    def __influence_matrix(self):
        return self.adjacency.matrix_transposed_.dot(self.state_matrix_.matrix)

    def __check_negative_contagion_correlation(self, contagions_above_threshold_not_active):
        return contagions_above_threshold_not_active.size > 0