import numpy as np
import time
import ray
import traceback
import logging

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List, Dict

from agamoo_ray.repair import DefaultRepair
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
                 init_pop: Optional[np.ndarray] = None,
                 create_method: str = 'lhs'):
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
            create_method (str): Method of creating the first population array ('uniform', 'lhs').
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
        self.create_method = create_method

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
        delta_iter = 0

        lower_bounds, upper_bounds = None, None
        if self.objective.bounds is not None:
            bounds_arr = np.array(self.objective.bounds)
            lower_bounds = bounds_arr[:, 0]
            upper_bounds = bounds_arr[:, 1]

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
            'player_id': self.num,
            'nobj': obj_idx,
            'population': pop,
            'population_eval': pop_eval,
            'evaluation_counter': self.evaluation_counter,
            'iteration_delta': 0,
            'iter_flag': False
        })
        self.evaluation_counter = 0

        try:
            while True:
                # 1. Fetch Global State Snapshot (Non-blocking via RefHolder)
                snapshot_ref = ray.get(self.ref_holder.get_ref.remote())
                if snapshot_ref is None:
                    time.sleep(0.01)
                    continue

                global_state = ray.get(snapshot_ref)

                if global_state['env_version'] != self.env_version:
                    params = global_state['env_params']
                    params['env_version'] = global_state['env_version']
                    self.update_environment(**params)

                # Check for termination signal
                if global_state['stop_flag']:
                    if self.verbose:
                        logger.info(f"Player {self.num} received stop signal. Terminating loop.")
                    break

                use_obj_map = global_state.get('use_obj_map', False)
                tracker_idx = self.num if use_obj_map else obj_idx

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
                                pop, pop_eval, neval = self.step(pop, pop_eval, np.ones_like(pattern, dtype=bool), global_state)
                            else:
                                pop, pop_eval, neval = self.step(pop, pop_eval, pattern, global_state)
                        except Exception as e:
                            logger.error(f"Player {self.num} error in step(): {e}", exc_info=True)
                            traceback.print_exc()

                    self.iteration += 1
                    delta_iter += 1
                    self.evaluation_counter += neval

                    # 2. Synchronization and Heartbeat Logic
                    iters = global_state['iter_counters'].copy()
                    iters[tracker_idx] = self.iteration

                    iters_mask = np.zeros(len(iters), dtype=bool)
                    for i in range(len(iters)):
                        if iters_pop is None or iters_pop[i] < iters[i]:
                            iters_mask[i] = True

                    # 3. Global Storage Update Dispatch
                    # Transmit full payload if other players have progressed, else send a lightweight heartbeat
                    if np.all(iters_mask[:obj_idx]) and np.all(iters_mask[obj_idx + 1:]):
                        ray.get(self.storage.update.remote({
                            'player_id': self.num,
                            'nobj': obj_idx,
                            'population': pop.copy(),
                            'population_eval': pop_eval.copy(),
                            'evaluation_counter': self.evaluation_counter,
                            'iteration_delta': delta_iter,
                            'iter_flag': False
                        }, env_version=self.env_version))
                        if self.verbose:
                            logger.info(f"Player {self.num} dispatched population update at iter {self.iteration}")

                        self.evaluation_counter = 0
                        next_iter_counter = 0
                        delta_iter = 0
                        iters_pop = iters.copy()
                    else:
                        # Heartbeat update (only iteration info)
                        self.storage.update.remote({
                            'player_id': self.num,
                            'nobj': obj_idx,
                            'iter_flag': True,
                            'iteration_delta': delta_iter
                        }, env_version=self.env_version)
                        delta_iter = 0
                        # Yield execution briefly to avoid hammering the object store
                        time.sleep(0.001)


                    # --- Cooperative Coevolution Exchange Logic ---
                    # Integrating external knowledge into the local population's unassigned genes
                    front = global_state['front']
                    front_eval = global_state['front_eval']
                    #_, unique_indices = np.unique(front_eval, axis=0, return_index=True)
                   # front = front[unique_indices]
                   # front_eval = front_eval[unique_indices]


                    best = global_state['best']
                    exchange_iter = global_state['exchange_iter']

                    if (self.iteration % exchange_iter == 0) and (self.exchange != 'none'):
                        modified_mask = np.zeros(pop.shape[0], dtype=bool)
                        if ('front_random' in self.exchange) and (len(front) > 0):
                            proc = 100
                            se = self.exchange.split('_')
                            if (len(se) == 3) and (0 < int(se[2]) < 100):
                                proc = int(se[2])

                            target_size = int(pop.shape[0] * (proc / 100))
                            nn = min(target_size, front.shape[0])
                            # nn = pop.shape[0]
                            if nn > front.shape[0]:
                                inds = np.random.choice(front.shape[0], nn, replace=True)
                            else:
                                inds = np.random.choice(front.shape[0], nn, replace=False)

                            #inds = np.random.choice(front.shape[0], nn, replace=True)
                            for i in range(nn):
                                # Inject non-optimized genes from random Pareto front members
                                #pop[i, np.logical_not(pattern)] = front[inds[i], np.logical_not(pattern)]
                                pop[i, :] = front[inds[i], :]
                                modified_mask[i] = True
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
                                #mask = front_suppression(local_front, local_front_eval, target_size, mode='objectives')
                                mask = self._front_suppression_cd(local_front_eval, target_size)
                                local_front = local_front[mask]

                            if len(local_front) > 0:
                                nn = min(target_size, local_front.shape[0])
                                if nn > local_front.shape[0]:
                                    inds = np.random.choice(local_front.shape[0], nn, replace=True)
                                else:
                                    inds = np.random.choice(local_front.shape[0], nn, replace=False)
                                for i in range(nn):
                                    # pop[i, np.logical_not(pattern)] = local_front[inds[i], np.logical_not(pattern)]
                                    pop[i, :] = local_front[inds[i], :]
                                    modified_mask[i] = True

                        elif (self.exchange == 'original') and (best is not None):
                            for i in range(len(best)):
                                if (i != obj_idx) and (best[i] is not None):
                                    pop[:, patterns[i]] = best[i][patterns[i]]
                            modified_mask[:] = True

                        elif (self.exchange == 'cross_sbx') and (len(front) > 0):
                            eta = 15.0
                            n_pop, n_vars = pop.shape

                            p1 = pop
                            front_idx = np.random.choice(len(front), n_pop, replace=True)
                            p2 = front[front_idx]

                            # --- Cross Probability ---
                            do_cross_ind = np.random.rand(n_pop) <= 0.9 #cross_prob
                            do_cross_var = np.random.rand(n_pop, n_vars) <= 0.9 #var_prob
                            do_crossover = do_cross_ind[:, np.newaxis] & do_cross_var #& ~pattern

                            # --- Simulated Binary Crossover (SBX) Math ---
                            u = np.random.rand(n_pop, n_vars)
                            beta = np.zeros_like(u)

                            mask_leq_05 = u <= 0.5
                            mask_gt_05 = ~mask_leq_05

                            beta[mask_leq_05] = (2.0 * u[mask_leq_05]) ** (1.0 / (eta + 1.0))
                            beta[mask_gt_05] = (1.0 / (2.0 * (1.0 - u[mask_gt_05]))) ** (1.0 / (eta + 1.0))

                            c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                            c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

                            take_c1 = np.random.rand(n_pop, n_vars) <= 0.5
                            # take_c1 = np.random.rand(n_pop, 1) <= 0.5
                            selected_children = np.where(take_c1, c1, c2)

                            pop = np.where(do_crossover, selected_children, p1)

                            if lower_bounds is not None:
                                pop = np.clip(pop, lower_bounds, upper_bounds)

                            modified_individuals = np.any(do_crossover, axis=1)
                            modified_mask[modified_individuals] = True

                        elif (self.exchange == 'cross') and (len(front) > 0):
                            n_pop, n_vars = pop.shape

                            p1 = pop
                            front_idx = np.random.choice(len(front), n_pop, replace=True)
                            p2 = front[front_idx]

                            # --- Cross Probability ---
                            do_cross_ind = np.random.rand(n_pop) <= 0.9 #cross_prob
                            do_cross_var = np.random.rand(n_pop, n_vars) <= 0.9 #var_prob
                            do_crossover = do_cross_ind[:, np.newaxis] & do_cross_var #& ~pattern

                            # --- Arithmetic Crossover Math ---
                            # Draw an alpha weight [0, 1] for each crossed gene independently
                            alpha = np.random.rand(n_pop, n_vars)

                            # Calculate the exact point between Parent 1 and Parent 2
                            child = alpha * p1 + (1.0 - alpha) * p2

                            # Gene substitution
                            pop = np.where(do_crossover, child, p1)

                            if lower_bounds is not None:
                                pop = np.clip(pop, lower_bounds, upper_bounds)

                            modified_individuals = np.any(do_crossover, axis=1)
                            modified_mask[modified_individuals] = True

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
                                #mask = front_suppression(local_front, local_front_eval, nn, mode='objectives')
                                mask = self._front_suppression_cd(local_front_eval, nn)
                                local_front = local_front[mask]

                            if len(local_front) > 0:
                                actual_nn = min(nn, local_front.shape[0])
                                inds = np.random.choice(local_front.shape[0], actual_nn, replace=True)
                                for i in range(actual_nn):
                                    pop[limit_idx + i, np.logical_not(pattern)] = local_front[
                                        inds[i], np.logical_not(pattern)]
                            modified_mask[:] = True

                        # Final Repair & Evaluate post-exchange to guarantee valid solutions

                        if np.any(modified_mask):
                            repaired_subset = self.repair.do(pop[modified_mask])
                            pop[modified_mask] = repaired_subset
                            new_evals = self.objective.evaluate(repaired_subset)
                            pop_eval[modified_mask] = new_evals
                            self.evaluation_counter += np.sum(modified_mask)

                        #if self.exchange != 'none':
                        #    pop = self.repair.do(pop)
                        #    pop_eval = self.objective.evaluate(pop)
                        #    self.evaluation_counter += pop.shape[0]

                    next_iter_counter += 1




        except Exception as e:
            logger.error(f"Player {self.num} crashed: {e}", exc_info=True)
            traceback.print_exc()
        finally:
            if self.verbose:
                logger.info(f"Player {self.num} successfully exited.")

    @abstractmethod
    def step(self, pop: np.ndarray, pop_eval: np.ndarray, pattern: np.ndarray, global_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Abstract method defining the core evolutionary step (e.g., Clonal Selection, Mutation).
        Must be implemented by subclasses.

        Args:
            pop (np.ndarray): Current population matrix.
            pop_eval (np.ndarray): Evaluated objective values for the population.
            pattern (np.ndarray): Boolean mask indicating which decision variables this player can modify.
            global_state: new

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
        """
        Generates the initial population based on the selected initialization strategy.
        """
        if self.create_method == 'lhs':
            return self._create_population_lhs()
        else:
            return self._create_population_uniform()

    def _create_population_uniform(self) -> np.ndarray:
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
    @staticmethod
    def _front_suppression_cd(front_eval, max_front):
        """Calculates crowding distance to maintain diversity during front exchange."""
        n_points, n_objs = front_eval.shape
        distances = np.zeros(n_points)

        # Iterate over each objective independently
        for m in range(n_objs):
            sorted_indices = np.argsort(front_eval[:, m])
            f_min = front_eval[sorted_indices[0], m]
            f_max = front_eval[sorted_indices[-1], m]

            # Assign infinity to boundary points (extremes) to prevent their deletion
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Safeguard against division by zero (if the entire front collapses to a single point)
            if f_max - f_min < 1e-9:
                continue

            # Fast vectorized calculation of distance for interior points
            prev_vals = front_eval[sorted_indices[:-2], m]
            next_vals = front_eval[sorted_indices[2:], m]
            distances[sorted_indices[1:-1]] += (next_vals - prev_vals) / (f_max - f_min)

        best_indices = np.argsort(distances)[::-1]
        return best_indices[:max_front]

    def _create_population_lhs(self) -> np.ndarray:
        """
        Initializes the population using Latin Hypercube Sampling (LHS).
        Solves the issue of empty 'holes' and 'clusters' in the decision space
        by guaranteeing a uniform division of the domain for each dimension.
        """
        pop_size, n_vars = self.npop, self.objective.n_var
        samples = np.empty((pop_size, n_vars))

        # Generate a uniform hypercube
        for i in range(n_vars):
            # 1. Random permutation assigning an individual to a 'bin' (grid cell)
            bins = np.random.permutation(pop_size)
            # 2. Random micro-tuning (noise) inside the assigned bin
            offset = np.random.uniform(0.0, 1.0, size=pop_size)
            # 3. Collapse values into a single dimension and normalize to [0.0, 1.0]
            samples[:, i] = (bins + offset) / pop_size

        # Scale the generated [0, 1] hypercube to the actual problem bounds
        if hasattr(self.objective, 'bounds') and self.objective.bounds is not None:
            bounds_arr = np.array(self.objective.bounds)
            lower_bounds = bounds_arr[:, 0]
            upper_bounds = bounds_arr[:, 1]

            # Linear interpolation: Min + Normalized_Value * (Max - Min)
            samples = lower_bounds + samples * (upper_bounds - lower_bounds)

        return samples


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
        self.env_version = 0

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

        if 'env_version' in kwargs:
            self.env_version = kwargs.pop('env_version')

        for obj in self.objectives:
            if hasattr(obj, 'update_env'):
                obj.update_env(**kwargs)
