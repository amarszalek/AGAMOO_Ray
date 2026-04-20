from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

class Repair(ABC):
    """
    Abstract base class for solution repair mechanisms in the AGAMOO framework.

    Repair methods are utilized to enforce constraints (e.g., boundary conditions,
    design restrictions) on solutions immediately after they have been modified
    by evolutionary operators (such as mutation or crossover).

    Specific repair strategies should inherit from this class and implement the `do` method.
    """
    def __init__(self, args: Optional[Any] = None, verbose: bool = False):
        """
        Initializes the Repair mechanism.

        Args:
            args (Any, optional): Additional parameters required for the repair logic.
                WARNING: Must be fully serializable (pickleable) for Ray compatibility.
            verbose (bool): Enables detailed execution logging if set to True.
        """
        self.args = args
        self.verbose = verbose

    @abstractmethod
    def do(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the specific repair logic to a population of solutions.

        Args:
            x (np.ndarray): A 2D NumPy array representing the population of solutions.

        Returns:
            np.ndarray: A 2D NumPy array containing the repaired (valid) solutions.
        """
        raise NotImplementedError('Subclasses must implement the do() method.')


class DefaultRepair(Repair):
    """
    Default repair mechanism that performs no modifications.
    Acts as a pass-through when no specific constraint handling is required
    for the given optimization problem.
    """
    def do(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the population unaltered.

        Args:
            x (np.ndarray): A 2D NumPy array of solutions.

        Returns:
            np.ndarray: The exact same unaltered 2D NumPy array.
        """
        return x
