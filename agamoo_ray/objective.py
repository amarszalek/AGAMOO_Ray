from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional
import numpy as np
import time


class Objective(ABC):
    """
    Abstract base class for optimization objectives used in the AGAMOO framework.

    In the distributed Ray architecture, instances of this class are serialized
    and transmitted to Player Actors. Therefore, the 'evaluate' method executes
    locally within the Player's isolated process, ensuring minimal overhead and
    high performance without global locks.

    Custom objective functions should inherit from this class and implement
    the abstract `evaluate` method.
    """
    def __init__(self,
                 num: int,
                 n_var: int,
                 n_obj: int,
                 bounds: List[Tuple[float, float]],
                 obj: int,
                 args: Optional[Any] = None,
                 verbose: bool = False):
        """
        Initializes the Objective instance.

        Args:
            num (int): Unique identifier for this objective.
            n_var (int): Total number of decision variables in the problem.
            n_obj (int): Total number of objective functions in the problem.
            bounds (List[Tuple[float, float]]): Lower and upper bounds for each decision variable.
            obj (int): The specific index of this objective function (0-based).
            args (Any, optional): Additional parameters required for evaluation.
                WARNING: Must be fully serializable (pickleable) for Ray compatibility.
            verbose (bool): Enables detailed logging if set to True.
        """
        self.num = num
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.obj = obj
        self.args = args
        self.verbose = verbose

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the objective function for a given population of solutions.

        Args:
            x (np.ndarray): A 2D NumPy array of shape (population_size, n_var)
                representing the decision variables of the solutions.

        Returns:
            np.ndarray: A 1D NumPy array of shape (population_size,) containing
                the evaluated objective values for each solution.
        """
        raise NotImplementedError('Subclasses must implement the evaluate() method.')

    def update_env(self, **kwargs) -> None:
        """
        Updates the dynamic parameters of the problem (e.g., time 'tau' for FDA problems).
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Force update for custom dynamic properties
                self.__dict__[key] = value

class TimedObjective(Objective):
    """
    Owija dowolne Objective i liczy, ile punktów oceniło i ile to trwało.
    Opcjonalnie dodaje opóźnienie na punkt (emulacja drogiego kryterium):
    busy=False -> time.sleep (nie obciąża CPU), busy=True -> aktywne czekanie (obciąża rdzeń
    jak prawdziwa symulacja).
    """
    def __init__(self, inner: Objective, delay_per_point: float = 0.0, busy: bool = False):
        super().__init__(inner.num, inner.n_var, inner.n_obj, inner.bounds, inner.obj, inner.args, inner.verbose)
        self.inner = inner
        self.delay = delay_per_point
        self.busy = busy
        self.calls = 0
        self.points = 0
        self.seconds = 0.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.inner.evaluate(x)
        if self.delay > 0:
            wait = self.delay * len(x)
            if self.busy:
                t_end = time.perf_counter() + wait
                while time.perf_counter() < t_end:
                    pass
            else:
                time.sleep(wait)
        self.calls += 1
        self.points += len(x)
        self.seconds += time.perf_counter() - t0
        return out

    def update_env(self, **kwargs) -> None:
        self.inner.update_env(**kwargs)


