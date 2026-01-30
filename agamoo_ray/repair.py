from abc import ABC, abstractmethod


class Repair(ABC):
    """
        Base class for repair methods used in AGAMO.
        Each repair should inherit from this class and implement the do() method.
    """
    def __init__(self, args=None, verbose=False):
        self.args = args
        self.verbose = verbose

    @abstractmethod
    def do(self, x):
        """Abstract method. Override this method by formula of repair method.

        Parameters
        ----------
        x : numpy.ndarray
            A 2-d numpy array of solutions.

        Returns
        -------
        y : numpy.ndarray
            A 2-d numpy array of solutions.
        """
        raise NotImplementedError('You must override this method in your class!')


class DefaultRepair(Repair):
    def do(self, x):
        return x
