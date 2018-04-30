from abc import abstractmethod

class BaseSingleContagionDiffusionModel:
    """
    # TODO Finish docstring
    Base class for multi-contagion diffusion models. Each class inheriting class should have :mathod: fit and
    :method: predict methods.
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
        Base method for prediction of information diffusion in multi-contagion world. It evaluates model's specific
        prediction methods.
        """
        pass