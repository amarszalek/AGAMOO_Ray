import numpy as np
from typing import Any, Optional, List, Tuple
from agamoo_ray import Objective

# Import type hint for Pymoo Problem if available, else use Any
try:
    from pymoo.core.problem import Problem as PymooProblem
except ImportError:
    PymooProblem = Any


class ProblemPymoo(Objective):
    """
    Adapter class to integrate optimization problems from the Pymoo library
    into the AGAMOO_Ray asynchronous framework.

    This wrapper maps Pymoo's centralized evaluation logic to AGAMOO's
    decentralized, agent-based architecture, where each Player Actor
    manages a single objective index.
    """

    def __init__(self,
                 num: int,
                 obj: int,
                 pymoo_prob: PymooProblem,
                 args: Optional[Any] = None,
                 verbose: bool = False):
        """
        Initializes the Pymoo problem adapter.

        Args:
            num (int): Unique identifier for the objective instance.
            obj (int): The objective index (1-based, consistent with AGAMOO's
                standard convention, internally converted to 0-based).
            pymoo_prob (PymooProblem): An instance of a Pymoo Problem class.
            args (Any, optional): Additional arguments for the evaluation.
            verbose (bool): Enables detailed logging if True.
        """
        # Internal conversion: users provide 1-based index (f1, f2...),
        # while Objective base class uses 0-based indexing.
        obj_idx = obj - 1

        # Extract metadata directly from the Pymoo problem instance
        n_var = pymoo_prob.n_var
        n_obj = pymoo_prob.n_obj
        self.prob = pymoo_prob

        # Map variable boundaries (xl: lower, xu: upper) to AGAMOO's list of tuples format
        bounds = list(zip(self.prob.xl, self.prob.xu))

        super().__init__(
            num=num,
            n_var=n_var,
            n_obj=n_obj,
            bounds=bounds,
            obj=obj_idx,
            args=args,
            verbose=verbose
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the specific objective function for the given population.

        Note: Pymoo typically evaluates all objectives simultaneously. This
        wrapper extracts only the relevant criterion for the current Player.

        Args:
            x (np.ndarray): 2D NumPy array of solutions (population).

        Returns:
            np.ndarray: 1D NumPy array of objective values for the assigned index.
        """
        # Pymoo returns a 2D array [population_size, n_obj]
        res = self.prob.evaluate(x)

        # Return only the column corresponding to this Player's objective
        return res[:, self.obj]

    def update_env(self, **kwargs) -> None:
        """
        Updates the dynamic parameters of the encapsulated Pymoo problem.
        """
        for key, value in kwargs.items():
            if hasattr(self.prob, key):
                setattr(self.prob, key, value)
            else:
                self.prob.__dict__[key] = value
