from abc import ABC, abstractmethod


class Objective(ABC):
    """
    Base class for optimization objectives used in AGAMOO (Ray version).

    In the Ray architecture, instances of this class are serialized and sent
    to Player Actors. Therefore, the 'evaluate' method runs locally within
    the Player's process, ensuring high performance.

    Each objective should inherit from this class and implement the evaluate() method.
    """
    def __init__(self, num: int, n_var: int, n_obj: int, bounds: list, obj: int, args=None, verbose=False):
        self.num = num
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.obj = obj
        self.args = args  # Warning: Must be serializable (pickleable) for Ray
        self.verbose = verbose

    @abstractmethod
    def evaluate(self, x):
        """Abstract method. Override this method by formula of objective function.

        Parameters
        ----------
        x : numpy.ndarray
            A 2-d numpy array of solutions.

        Returns
        -------
        y : numpy.ndarray
            A 1-d numpy array of values of objective function for given x.
        """
        raise NotImplementedError('You must override this method in your class!')
