from abc import abstractmethod


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
