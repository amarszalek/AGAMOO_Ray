import numpy as np
import time
import ray
import traceback
import logging

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List

from agamoo_ray.repair import DefaultRepair
from agamoo_ray.utils import front_suppression
from agamoo_ray.objective import Objective

logger = logging.getLogger(__name__)


class Player(ABC):
    """
    Abstract base class for an autonomous 'Player' entity in the AGAMOO framework.

    In the Ray-based architecture, concrete implementations of this class
    (e.g., ClonalSelection) should be decorated with @ray.remote to operate
    as independent, asynchronous Actors representing specific optimization objectives.
    """

    def __init__(self,
                 num: int,
                 npop: int,
                 objective: Objective,
                 storage_actor: Any,
                 gens: str ='pattern',
                 exchange: str ='front_sup',
                 verbose: bool =False,
                 init_pop: Optional[np.ndarray] = None):
        """
        Initializes the Player entity.

        Args:
            num (int): Unique identifier index for the player.
            npop (int): Population size managed by this specific player.
            objective (Objective): The specific objective function this player aims to optimize.
            storage_actor (Any): Handle to the GlobalStorage Ray Actor.
            gens (str): Gene allocation strategy ('pattern' or 'all').
            exchange (str): Gene exchange strategy ('mix', 'original', 'front_random', 'front_sup').
            verbose (bool): Enables detailed execution logging.
            init_pop (np.ndarray, optional): Custom initial population array.
        """
        self.num = num
        self.npop = npop
        self.objective = objective
        self.storage = storage_actor
        self.gens = gens
        self.exchange = exchange
        self.verbose = verbose
        self.repair = DefaultRepair()
        self.init_pop = init_pop

        self.env_version = 0

        self.ref_holder: Optional[Any] = None

        self.iteration: int = 0
        self.evaluation_counter: int = 0

    def set_repair(self, repair: Any) -> None:
        """Assigns a custom repair mechanism for out-of-bounds solutions."""
        if repair is not None:
            self.repair = repair

    def set_infrastructure(self, storage: Any, ref_holder: Any) -> None:
        """
        Dependency injection for distributed Ray components.

        Args:
            storage (Any): Handle to the GlobalStorage Actor.
            ref_holder (Any): Handle to the RefHolder Actor for non-blocking reads.
        """
        self.storage = storage
        self.ref_holder = ref_holder

    def start(self) -> None:
        """
        Main execution loop of the player. When deployed via Ray, this runs
        continuously inside the Actor's dedicated process, enabling true asynchronous
        optimization without global barriers.
        """
        if self.verbose:
            logger.info(f"Player {self.num} started (Ray Actor).")

        obj_idx = self.objective.obj
        next_iter_counter = 0
        iters_pop: Optional[np.ndarray] = None

        # Local population initialization
        if self.init_pop is not None:
            pop = self.init_pop.copy()
        else:
            pop = self.create_population()

        # Initial Evaluation & Repair
        pop = self.repair.do(pop)
        pop_eval = self.objective.evaluate(pop)
        self.evaluation_counter += pop.shape[0]

        if self.verbose:
            logger.info(f"Player {self.num} evaluated the initial population.")

        # Dispatch initial population to the Global Storage asynchronously
        self.storage.update.remote({
            'nobj': obj_idx,
            'population': pop,
            'population_eval': pop_eval,
            'evaluation_counter': self.evaluation_counter,
            'iteration': 0,
            'iter_flag': False
        })

        try:
            while True:
                # 1. Fetch Global State Snapshot (Non-blocking via RefHolder)
                snapshot_ref = ray.get(self.ref_holder.get_ref.remote())
                if snapshot_ref is None:
                    time.sleep(0.01)
                    continue

                global_state = ray.get(snapshot_ref)

                # Check for termination signal
                if global_state['stop_flag']:
                    if self.verbose:
                        logger.info(f"Player {self.num} received stop signal. Terminating loop.")
                    break

                # Retrieve current locus assignment (DVA mechanism)
                patterns = global_state['patterns']
                pattern = patterns[obj_idx]
                next_iter = global_state['next_iter']

                if next_iter <= 0 or next_iter - next_iter_counter > 0:
                    neval = 0

                    if pattern.sum() > 0:
                        try:
                            # Execute optimization step only on assigned variables
                            if self.gens == 'all':
                                pop, pop_eval, neval = self.step(pop, pop_eval, np.ones_like(pattern, dtype=bool))
                            else:
                                pop, pop_eval, neval = self.step(pop, pop_eval, pattern)
                        except Exception as e:
                            logger.error(f"Player {self.num} error in step(): {e}", exc_info=True)
                            traceback.print_exc()

                    self.iteration += 1
                    self.evaluation_counter += neval

                    # --- Cooperative Coevolution Exchange Logic ---
                    # Integrating external knowledge into the local population's unassigned genes
                    front = global_state['front']
                    front_eval = global_state['front_eval']
                    best = global_state['best']

                    if (self.exchange == 'front_random') and (len(front) > 0):
                        nn = pop.shape[0]
                        inds = np.random.choice(front.shape[0], nn, replace=True)
                        for i in range(nn):
                            # Inject non-optimized genes from random Pareto front members
                            pop[i, np.logical_not(pattern)] = front[inds[i], np.logical_not(pattern)]

                    elif ('front_sup' in self.exchange) and (len(front) > 0):
                        proc = 100
                        se = self.exchange.split('_')
                        if (len(se) == 3) and (0 < int(se[2]) < 100):
                            proc = int(se[2])

                        arr = np.arange(front.shape[0])
                        np.random.shuffle(arr)
                        local_front = front[arr]
                        local_front_eval = front_eval[arr]

                        # Apply distance suppression to maintain diversity during exchange
                        target_size = int(pop.shape[0] * (proc / 100))
                        if target_size < local_front.shape[0]:
                            mask = front_suppression(local_front_eval, target_size)
                            local_front = local_front[mask]

                        if len(local_front) > 0:
                            nn = min(target_size, local_front.shape[0])
                            inds = np.random.choice(local_front.shape[0], nn, replace=True)
                            for i in range(nn):
                                pop[i, np.logical_not(pattern)] = local_front[inds[i], np.logical_not(pattern)]

                    elif (self.exchange == 'original') and (best is not None):
                        for i in range(len(best)):
                            if (i != obj_idx) and (best[i] is not None):
                                pop[:, patterns[i]] = best[i][patterns[i]]

                    elif ('mix' in self.exchange) and (best is not None) and (len(front) > 0):
                        proc = 50
                        se = self.exchange.split('_')
                        if (len(se) == 2) and (0 < int(se[1]) < 100):
                            proc = int(se[1])

                        # Phase 1: Integrate genes from the specific best solutions
                        limit_idx = int(pop.shape[0] * (proc / 100))
                        for i in range(len(best)):
                            if (i != obj_idx) and (best[i] is not None):
                                pop[:limit_idx, patterns[i]] = best[i][patterns[i]]

                        # Phase 2: Integrate genes from the diverse Pareto front
                        arr = np.arange(front.shape[0])
                        np.random.shuffle(arr)
                        local_front = front[arr]
                        local_front_eval = front_eval[arr]

                        nn = (pop.shape[0] - limit_idx)
                        if nn < local_front.shape[0]:
                            mask = front_suppression(local_front_eval, nn)
                            local_front = local_front[mask]

                        if len(local_front) > 0:
                            actual_nn = min(nn, local_front.shape[0])
                            inds = np.random.choice(local_front.shape[0], actual_nn, replace=True)
                            for i in range(actual_nn):
                                pop[limit_idx + i, np.logical_not(pattern)] = local_front[
                                    inds[i], np.logical_not(pattern)]

                    # Final Repair & Evaluate post-exchange to guarantee valid solutions
                    pop = self.repair.do(pop)
                    pop_eval = self.objective.evaluate(pop)

                    self.evaluation_counter += pop.shape[0]
                    next_iter_counter += 1

                # 2. Synchronization and Heartbeat Logic
                iters = global_state['iter_counters'].copy()
                iters[obj_idx] = self.iteration

                iters_mask = np.zeros(len(iters), dtype=bool)
                for i in range(len(iters)):
                    if iters_pop is None or iters_pop[i] < iters[i]:
                        iters_mask[i] = True

                # 3. Global Storage Update Dispatch
                # Transmit full payload if other players have progressed, else send a lightweight heartbeat
                if np.all(iters_mask[:obj_idx]) and np.all(iters_mask[obj_idx + 1:]):
                    self.storage.update.remote({
                        'nobj': obj_idx,
                        'population': pop.copy(),
                        'population_eval': pop_eval.copy(),
                        'evaluation_counter': self.evaluation_counter,  # diff since last update
                        'iteration': self.iteration,
                        'iter_flag': False
                    }, env_version=self.env_version)
                    if self.verbose:
                        logger.info(f"Player {self.num} dispatched population update at iter {self.iteration}")

                    next_iter_counter = 0
                    iters_pop = iters.copy()
                else:
                    # Heartbeat update (only iteration info)
                    self.storage.update.remote({
                        'nobj': obj_idx,
                        'iter_flag': True,
                        'iteration': self.iteration
                    }, env_version=self.env_version)
                    # Yield execution briefly to avoid hammering the object store
                    time.sleep(0.001)

        except Exception as e:
            logger.error(f"Player {self.num} crashed: {e}", exc_info=True)
            traceback.print_exc()
        finally:
            if self.verbose:
                logger.info(f"Player {self.num} successfully exited.")

    @abstractmethod
    def step(self, pop: np.ndarray, pop_eval: np.ndarray, pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Abstract method defining the core evolutionary step (e.g., Clonal Selection, Mutation).
        Must be implemented by subclasses.

        Args:
            pop (np.ndarray): Current population matrix.
            pop_eval (np.ndarray): Evaluated objective values for the population.
            pattern (np.ndarray): Boolean mask indicating which decision variables this player can modify.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Updated population, updated evaluations, and number of evaluations performed.
        """
        raise NotImplementedError('Subclasses must override the step() method.')

    def evaluate(self, pop: np.ndarray) -> np.ndarray:
        """
        Helper method utilized by GlobalStorage to compute missing criteria
        for solutions generated by other players.
        """
        return self.objective.evaluate(pop)

    def update_environment(self, **kwargs) -> None:
        """Asynchronously updates the player's environment."""

        if 'env_version' in kwargs:
            self.env_version = kwargs.pop('env_version')

        if hasattr(self.objective, 'update_env'):
            self.objective.update_env(**kwargs)

    def create_population(self) -> np.ndarray:
        """Generates the initial population using uniform distribution across variable bounds."""
        pop = np.zeros((self.npop, self.objective.n_var))
        for i in range(self.npop):
            pop[i] = self._create_individual_uniform(self.objective.bounds)
        return pop

    @staticmethod
    def _create_individual_uniform(bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Creates a single individual within the defined problem boundaries."""
        a = np.array([bounds[k][0] for k in range(len(bounds))])
        b = np.array([bounds[k][1] for k in range(len(bounds))])
        return np.random.uniform(a, b)


@ray.remote
class Evaluator:
    """
    Dedicated Ray Actor for computing objective functions.
    Acts as an asynchronous 'calculator node' for the GlobalStorage to balance evaluation loads.
    """
    def __init__(self, objectives: List[Objective]):
        """
        Args:
            objectives (List[Objective]): List of Objective instances to evaluate against.
        """
        self.objectives = objectives

    def evaluate(self, pop: np.ndarray, i: int) -> np.ndarray:
        """
        Evaluates a population against a specific objective index.

        Args:
            pop (np.ndarray): The population to evaluate.
            i (int): Index of the objective function.

        Returns:
            np.ndarray: Flattened array of evaluation results.
        """
        res = self.objectives[i].evaluate(pop)
        return np.array(res).flatten()

    def update_environment(self, **kwargs) -> None:
        """Asynchronously updates the evaluator's environment."""
        for obj in self.objectives:
            if hasattr(obj, 'update_env'):
                obj.update_env(**kwargs)
