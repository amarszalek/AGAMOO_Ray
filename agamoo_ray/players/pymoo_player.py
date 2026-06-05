import numpy as np
import ray
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional, Type
import logging

try:
    from pymoo.core.problem import Problem
    from pymoo.core.population import Population
    from pymoo.core.termination import NoTermination
    from pymoo.core.algorithm import Algorithm
except ImportError:
    # Fallback for typing if pymoo is not installed
    Problem, Population, NoTermination, Algorithm = Any, Any, Any, Any

from agamoo_ray.player import Player
from agamoo_ray.objective import Objective

logger = logging.getLogger(__name__)


@ray.remote
class PymooPlayer(Player):
    """
    Asynchronous Ray Actor that wraps any optimization algorithm from the Pymoo library
    (e.g., GA, DE, PSO) to act as a single-objective Player within the AGAMOO framework.
    """

    def __init__(self,
                 num: int,
                 npop: int,
                 objective: Objective,
                 storage_actor: Any,
                 pymoo_alg_cls: Type[Algorithm],
                 alg_kwargs: Optional[Dict[str, Any]] = None,
                 gens: str = 'pattern',
                 exchange: str = 'front_sup',
                 verbose: bool = False,
                 init_pop: Optional[np.ndarray] = None):
        """
        Args:
            pymoo_alg_cls: The class reference of the Pymoo algorithm (e.g., GA, PSO).
            alg_kwargs: Dictionary of hyperparameters to pass to the Pymoo algorithm.
        """
        super().__init__(num, npop, objective, storage_actor, gens, exchange, verbose, init_pop)

        self.pymoo_alg_cls = pymoo_alg_cls
        self.alg_kwargs = alg_kwargs or {}

        # State tracking for Pymoo Algorithm and dynamic DVA patterns
        self.current_pattern: Optional[np.ndarray] = None
        self.alg: Optional[Algorithm] = None
        self.sub_problem: Optional[Problem] = None

    def step(self, pop: np.ndarray, pop_eval: np.ndarray, pattern: np.ndarray, global_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        s = np.sum(pattern)
        if s == 0:
            return pop, pop_eval, 0

        indx = np.where(pattern)[0]

        # 1. State Management: Reinitialize ONLY if the AAL mechanism rotated the assigned genes
        if self.current_pattern is None or not np.array_equal(self.current_pattern, pattern):
            self.current_pattern = pattern.copy()

            # Create a dynamic sub-problem bounds for the active genes
            xl = np.array([self.objective.bounds[k][0] for k in indx])
            xu = np.array([self.objective.bounds[k][1] for k in indx])
            self.sub_problem = Problem(n_var=s, n_obj=1, n_constr=0, xl=xl, xu=xu)

            # Setup Pymoo algorithm
            kwargs = self.alg_kwargs.copy()
            kwargs['pop_size'] = pop.shape[0]

            self.alg = self.pymoo_alg_cls(**kwargs)
            self.alg.setup(self.sub_problem, termination=NoTermination())

        # 2. Sync Pymoo's internal state with AGAMOO
        # AGAMOO's exchange logic (Cooperative Coevolution) might have modified the population externally.
        # We must push this updated knowledge into Pymoo before asking for new offspring.
        pymoo_pop = Population.new("X", pop[:, indx])
        pymoo_pop.set("F", pop_eval.reshape(-1, 1))
        self.alg.pop = pymoo_pop

        # 3. Pymoo 'Ask': Generate new candidates (offspring) for active genes
        infills = self.alg.ask()
        infills_x = infills.get("X")

        # 4. Reconstruct full individuals to evaluate them
        # We take parent structures and replace ONLY the active genes with Pymoo's offspring
        n_offspring = len(infills_x)
        full_offspring = np.zeros((n_offspring, pop.shape[1]))

        parent_indices = np.arange(n_offspring) % pop.shape[0]
        full_offspring[:, :] = pop[parent_indices, :]
        full_offspring[:, indx] = infills_x

        # 5. Evaluate the newly constructed individuals natively in AGAMOO
        full_offspring = self.repair.do(full_offspring)
        offspring_eval = self.objective.evaluate(full_offspring)
        evals_count = n_offspring

        # 6. Pymoo 'Tell': Return evaluated offspring to Pymoo
        # This triggers Pymoo's internal Survival mechanism (Selection for the next generation)
        infills.set("F", offspring_eval.reshape(-1, 1))
        self.alg.tell(infills=infills)

        # 7. Extract the survived population from Pymoo's internal state
        survivors_x = self.alg.pop.get("X")
        survivors_evals = self.alg.pop.get("F").flatten()

        # 8. Reconstruct the final population payload for the AGAMOO Global Storage
        n_survivors = len(survivors_x)
        new_pop = np.zeros((n_survivors, pop.shape[1]))

        parent_indices2 = np.arange(n_survivors) % pop.shape[0]
        new_pop[:, :] = pop[parent_indices2, :]
        new_pop[:, indx] = survivors_x

        return new_pop, survivors_evals, evals_count