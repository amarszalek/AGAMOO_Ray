import time
import ray
import numpy as np
import logging
import asyncio
import traceback
import pickle
from copy import deepcopy
from tqdm.auto import tqdm
from agamoo_ray.utils import get_not_dominated, front_suppression
from agamoo_ray.utils import assigning_gens, adaptive_linear_assigning_gens, adaptive_shap_assigning_gens, adaptive_sparsity_gens
from typing import List, Dict, Any, Optional, Callable, Union


logger = logging.getLogger(__name__)


class AGAMOO:
    """
        Main driver class for the Asynchronous Game Theory Multi-objective
        Optimization (AGAMOO) framework using the Ray actor model.
    """
    def __init__(self,
                 max_eval: int,
                 change_iter: int,
                 exchange_iter: int,
                 next_iter: int,
                 max_front: int,
                 max_front_tol: float = 0.0,
                 init_pop: str = 'separate',
                 assign_gens: str = 'random',
                 front_f: Optional[Callable] = None,
                 sup_mode: str = 'objectives',
                 verbose: bool = False,
                 log_freq: int = 0):

        """
        Initializes the AGAMOO framework.

        Args:
            max_eval (int): Maximum number of objective function evaluations.
            change_iter (int): Number of iterations before executing the dynamic variable assignment (DVA).
            next_iter (int): Base iterations for the player cycle.
            max_front (int): Maximum size of the global Pareto front archive.
            max_front_tol (float): Tolerance for the Pareto front size before suppression triggers.
            init_pop (str): Strategy for initial population generation.
            assign_gens (str): Strategy for locus allocation ('random' or 'adaptive_linear').
            front_f (Callable, optional): Custom filtering function for the Pareto front.
            verbose (bool): Enables detailed logging if True.
            log_freq (int): Frequency of logging the global state for convergence analysis.
        """

        self.max_eval = max_eval
        self.change_iter = change_iter
        self.exchange_iter = exchange_iter
        self.next_iter = next_iter
        self.max_front = max_front
        self.max_front_tol = max_front_tol
        self.init_pop = init_pop
        self.front_f = front_f
        self.verbose = verbose
        self.assign_gens = assign_gens
        self.log_freq = log_freq
        self.sup_mode = sup_mode

        self.env_version = 0

        self.players: List[Any] = []
        self.evaluators: List[Any] = []
        self.storage: Optional[Any] = None
        self.results: Optional[Dict] = None
        self.ref_holder: Optional[Any] = None
        self.repair: Optional[Any] = None

        self.nobjs: int = 0
        self.nvars: int = 0

    def init_players(self, players: List[Any], evaluator: Union[List[Any], Any], repair: Optional[Any] = None) -> None:
        """
        Registers the Player and Evaluator Ray actors within the framework.

        Args:
            players (List[ray.actor.ActorHandle]): List of Player actors.
            evaluator (Union[List[ray.actor.ActorHandle], ray.actor.ActorHandle]): Evaluator actor(s).
            repair (Any, optional): Mechanism for repairing out-of-bounds solutions.
        """
        self.players = players
        self.evaluators = evaluator if isinstance(evaluator, list) else [evaluator]
        self.repair = repair

        # Register components in the global storage
        ray.get(self.storage.set_players.remote(players))
        ray.get(self.storage.set_evaluator.remote(self.evaluators))

        for p in self.players:
            p.set_infrastructure.remote(self.storage, self.ref_holder)


    def create_storage(self, nvars: int, nobjs: int, num_cpus: int = 1) -> Any:
        """
        Initializes the Ray environment and creates the GlobalStorage and RefHolder actors.

        Args:
            nvars (int): Number of decision variables in the optimization problem.
            nobjs (int): Number of objective functions.
            num_cpus (int): Number of CPUs allocated for the storage actor.

        Returns:
            ray.actor.ActorHandle: Handle to the created GlobalStorage actor.
        """
        self.nvars = nvars
        self.nobjs = nobjs

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        self.ref_holder = RefHolder.remote()

        self.storage = GlobalStorage.options(num_cpus=num_cpus).remote(
            nvars, nobjs, self.max_eval, self.change_iter, self.exchange_iter, self.next_iter,
            self.max_front, self.assign_gens, self.max_front_tol, self.front_f, self.sup_mode,
            ref_holder=self.ref_holder, verbose=self.verbose, log_freq=self.log_freq,
        )
        return self.storage

    def start_optimize(self, tqdm_disable: bool = False, background: bool = False) -> None:
        """
        Starts the asynchronous optimization process and monitors the stop conditions.

        Args:
            tqdm_disable (bool): If True, disables the progress bar.
            background (bool): If True, starts optimization in the background and returns
                               immediately. If False, blocks until max_eval is reached.
        """
        if not self.players:
            raise ValueError("Players list is empty. Initialize players first.")
        if not self.storage:
            raise ValueError("Storage not created. Call create_storage() first.")

        # Reset global state for a fresh run
        ray.get(self.storage.reset.remote())

        if self.verbose:
            logger.info(f"Starting AGAMOO optimization ({'BACKGROUND' if background else 'BLOCKING'})...")

        # Start asynchronous player loops
        for p in self.players:
            p.set_repair.remote(self.repair)
            p.start.remote()

        if self.verbose:
            logger.info("Players are running in the background...")

        # Mode: Background execution
        if background:
            logger.info("Optimization is running in the background. Use model.stop() to terminate.")
            return

        # Driver loop: Monitoring the global state via non-blocking RefHolder
        try:
            with tqdm(total=self.max_eval, unit='eval', disable=tqdm_disable) as pbar:
                while True:
                    ref = ray.get(self.ref_holder.get_ref.remote())
                    if ref is None:
                        time.sleep(0.1)
                        continue

                    state = ray.get(ref)
                    current_evals = state.get('evaluations', 0)
                    stop_flag = state.get('stop_flag', False)

                    pbar.n = min(current_evals, self.max_eval)
                    pbar.refresh()

                    if stop_flag:
                        break

                    time.sleep(0.5)
        except KeyboardInterrupt:
            if self.verbose:
                logger.info("Optimization interrupted by user (KeyboardInterrupt).")
        finally:
            # Automatic cleanup in blocking mode
            if not background:
                self.stop()

    def stop(self) -> None:
        """
        Forcefully stops the optimization and terminates all Player actors.
        """
        if self.storage:
            # Set stop flag in GlobalStorage so actors can exit gracefully if possible
            ray.get(self.storage.force_stop.remote())

            # Kill Ray actors to free resources
            for p in self.players:
                ray.kill(p)

            if self.verbose:
                logger.info("AGAMOO optimization stopped and resources cleared.")

    def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current status (evaluations, iterations, stop_flag) from storage.
        """
        if not self.storage:
            return None
        return ray.get(self.storage.get_status.remote())

    def get_results(self, key: Optional[str] = None) -> Any:
        """Retrieves the final optimization results from the global storage."""
        if not self.storage:
            return None

        if key:
            ref = ray.get(self.ref_holder.get_ref.remote())
            if ref is None:
                return None
            snapshot = ray.get(ref)
            return snapshot.get(key)

        return ray.get(self.storage.get_results.remote())

    def get_history(self) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieves the full optimization history directly from the global storage.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing historical states,
                                  or None if the storage is not initialized.
        """
        if not self.storage:
            logger.warning("Storage not created. Cannot fetch history.")
            return None

        return ray.get(self.storage.get_history.remote())

    def update_environment(self, reevaluate_front: bool = True, **kwargs) -> None:
        """
        Broadcasts new environment parameters to players and evaluators,
        and then forces a re-evaluation of the points in the archive.
        """
        if self.verbose:
            logger.info(f"Broadcasting environment update: {kwargs}")

        self.env_version += 1
        env_params = kwargs.copy()
        kwargs['env_version'] = self.env_version

        # 1. Update all players (they operate independently in the background)
        #for p in self.players:
        #    if hasattr(p, 'update_environment'):
        #       p.update_environment.remote(**kwargs)

        # 2. Update evaluators and WAIT for confirmation
        # This is critical: Storage must use the updated evaluators to re-evaluate the archive
        update_futures = []
        for e in self.evaluators:
            update_futures.append(e.update_environment.remote(**kwargs))

        if update_futures:
            ray.get(update_futures)  # Block for a fraction of a second until evaluators are updated

        # 3. Dispatch archive re-evaluation (GlobalStorage will use the updated evaluators)
        if reevaluate_front and self.storage:
            ray.get(self.storage.reevaluate_archive.remote(env_version=self.env_version, env_params=env_params))


@ray.remote
class GlobalStorage:
    """
    Ray actor responsible for maintaining the global state, Pareto front archive,
    and handling the Dynamic Variable Assignment mechanisms.
    """

    def __init__(self,
                 nvars: int,
                 nobjs: int,
                 max_eval: int,
                 change_iter: int,
                 exchange_iter: int,
                 next_iter: int,
                 max_front: int,
                 assign_gens: str = 'random',
                 max_front_tol: float = 0.0,
                 front_f: Optional[Callable] = None,
                 sup_mode: str = 'objectives',
                 verbose: bool = False,
                 ref_holder: Optional[Any] = None,
                 log_freq=0):

        self.nvars = nvars
        self.nobjs = nobjs
        self.max_eval = max_eval
        self.change_iter = change_iter
        self.exchange_iter = exchange_iter
        self.next_iter = next_iter
        self.max_front = max_front
        self.max_front_tol = max_front_tol
        self.front_f = front_f
        self.sup_mode = sup_mode

        self.current_env_version = 0
        self.current_env_params = {}

        self.players_handles: List[Any] = []
        self.evaluators: List[Any] = []
        self.eval_rr_index: int = 0  # Round-Robin index for load balancing
        self.verbose = verbose
        self.assign_gens = assign_gens

        self.ref_holder = ref_holder

        self.start_time = time.perf_counter_ns()
        self.log_freq = log_freq
        self.last_logged_iter = 0

        self.history: List[Dict] = []
        self.lpatterns: List[np.ndarray] = []

        # Internal state initialization
        self.reset()
        self.log_current_state(0)

    def set_players(self, players: List[Any]) -> None:
        """Registers player actor handles."""
        self.players_handles = players

    def set_evaluator(self, evaluators: Union[List[Any], Any]) -> None:
        """Registers evaluator actor handles."""
        self.evaluators = evaluators if isinstance(evaluators, list) else [evaluators]

    def _refresh_snapshot_ref(self) -> None:
        """
        Creates a snapshot of the global state in the Ray Plasma Store and
        asynchronously passes the reference to the RefHolder.
        """
        snapshot_data = {
            'front': self.front,
            'front_eval': self.front_eval,
            'best': self.best,
            'iter_counters': self.iter_counters,
            'patterns': self.patterns,
            'next_iter': self.next_iter,
            'exchange_iter': self.exchange_iter,
            'stop_flag': self.stop_flag,
            'evaluations': self.total_evaluations,
            'env_version': self.current_env_version,
            'env_params': self.current_env_params
        }
        # ray.put stores data in shared memory and returns a lightweight ObjectRef
        ref = ray.put(snapshot_data)

        # Asynchronous, non-blocking signal to RefHolder
        self.ref_holder.update_ref.remote([ref])

    def reset(self) -> None:
        """Resets the internal state before a new optimization run."""
        self.front = np.empty((0, self.nvars))
        self.front_eval = np.empty((0, self.nobjs))
        self.best = [None] * self.nobjs

        self.iter_counters = np.zeros(self.nobjs)
        self.evaluations_count = np.zeros(self.nobjs)
        self.evaluations_time = np.zeros(self.nobjs)
        self.repair_time = np.zeros(self.nobjs)

        self.stop_flag = False
        self.min_iter_pop = 0

        # Initialize variable patterns (genes)
        self.patterns = assigning_gens(self.nvars, self.nobjs)
        self.total_evaluations = 0

        self._refresh_snapshot_ref()

    def force_stop(self) -> None:
        """Sets the internal stop flag to True."""
        self.stop_flag = True
        self._refresh_snapshot_ref()

    def get_status(self) -> Dict[str, Any]:
        """Returns the current optimization status."""
        return {
            'iterations': self.iter_counters,
            'total_evaluations': self.total_evaluations,
            'evaluations': self.evaluations_count,
            'stop_flag': self.stop_flag,
            'front_size': len(self.front)
        }

    def get_results(self) -> Dict[str, Any]:
        """Returns the final, suppressed Pareto front and evaluations stats."""
        final_front = self.front
        final_front_eval = self.front_eval

        if len(final_front) > self.max_front:
            mask = front_suppression(final_front, final_front_eval, self.max_front, mode=self.sup_mode)
            final_front = final_front[mask]
            final_front_eval = final_front_eval[mask]

        return {
            'front': final_front,
            'front_eval': final_front_eval,
            'iter_counters': self.iter_counters,
            'evaluations': self.evaluations_count
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Returns the in-memory optimization history recorded so far.
        Useful for live tracking or notebook visualizations without saving to disk.
        """
        return self.history

    async def update(self, data: Dict[str, Any], env_version: int = 0) -> None:
        """
        Main asynchronous method handling updates from Player actors.
        Updates the global Pareto archive and evaluates missing criteria.
        """

        try:
            nobj = data['nobj']
            iteration = data.get('iteration', 0)

            if self.verbose:
                logger.info(f"GlobalStorage received update from objective {nobj}")

            self.iter_counters[nobj] = iteration

            # Return early if it's just a heartbeat
            if data.get('iter_flag', False):
                self._refresh_snapshot_ref()
                return

            if env_version < self.current_env_version:
                return

            # Dynamic Variable Assignment logic
            min_iter = np.min(self.iter_counters)
            if min_iter - self.min_iter_pop >= self.change_iter:
                if self.assign_gens=='random':
                    self.patterns = assigning_gens(self.nvars, self.nobjs)
                elif self.assign_gens=='adaptive_linear':
                    self.patterns = adaptive_linear_assigning_gens(self.front, self.front_eval, self.nvars, self.nobjs)
                elif self.assign_gens == 'adaptive_sparsity':
                    self.patterns = adaptive_sparsity_gens(self.front, self.front_eval, self.nvars, self.nobjs)
                elif self.assign_gens=='adaptive_shap':
                    self.patterns = adaptive_shap_assigning_gens(self.front, self.front_eval, self.nvars, self.nobjs)
                else:
                    raise ValueError(f"Unknown assign_gens strategy: {self.assign_gens}")
                self.min_iter_pop = min_iter

            self.lpatterns.append(self.patterns.copy())

            # Extract population data
            pop = data['population']
            pop_eval_partial = data['population_eval']
            neval = data['evaluation_counter']

            if neval > 0:
                self.evaluations_count[nobj] += neval
            self.total_evaluations = np.min(self.evaluations_count)

            # Update the Best Solution for the corresponding objective
            if len(pop_eval_partial) > 0:
                best_idx = np.argmin(pop_eval_partial)
                self.best[nobj] = pop[best_idx].copy()

            pop_eval = np.zeros((pop.shape[0], self.nobjs))
            pop_eval[:, nobj] = pop_eval_partial

            futures = []
            target_objs = []
            num_workers = len(self.evaluators)

            # Evaluate the population on remaining objectives
            for i in range(self.nobjs):
                if i != nobj and num_workers > 0:
                    evaluator = self.evaluators[self.eval_rr_index % num_workers]
                    self.eval_rr_index += 1
                    futures.append(evaluator.evaluate.remote(pop, i))
                    target_objs.append(i)

            # Gather results asynchronously
            if futures:
                results = await asyncio.gather(*futures)
                for idx, res in enumerate(results):
                    obj_idx = target_objs[idx]
                    pop_eval[:, obj_idx] = res
                    self.evaluations_count[obj_idx] += pop.shape[0]

            self.total_evaluations = np.min(self.evaluations_count)

            # Merge with the global Pareto front
            if len(self.front) == 0:
                self.front = pop
                self.front_eval = pop_eval
            else:
                self.front = np.vstack([self.front, pop])
                self.front_eval = np.vstack([self.front_eval, pop_eval])

            _, unique_indices = np.unique(self.front, axis=0, return_index=True)
            self.front = self.front[unique_indices]
            self.front_eval = self.front_eval[unique_indices]

            # Non-dominated selection
            if len(self.front) > 1:
                mask = get_not_dominated(self.front_eval)
                self.front = self.front[mask]
                self.front_eval = self.front_eval[mask]

            # Custom user filtering (front_f)
            if self.front_f is not None and len(self.front) > 0:
                mask = self.front_f(self.front_eval)
                self.front = self.front[mask]
                self.front_eval = self.front_eval[mask]

            # Archive suppression (Size limit)
            limit = int(self.max_front * (1.0 + self.max_front_tol)) if self.max_front_tol > 0 else self.max_front

            if len(self.front) > limit:
                mask = front_suppression(self.front, self.front_eval, self.max_front, mode=self.sup_mode)
                self.front = self.front[mask]
                self.front_eval = self.front_eval[mask]

            # Stop condition check
            if self.max_eval > 0 and self.total_evaluations >= self.max_eval:
                self.stop_flag = True

            self._refresh_snapshot_ref()

            # Logging logic
            if self.log_freq > 0 and (min_iter >= self.last_logged_iter + self.log_freq):
                self.log_current_state(min_iter)

        except Exception as e:
            logger.error(f"GlobalStorage update error: {e}", exc_info=True)
            traceback.print_exc()

    def log_current_state(self, current_iter: int) -> None:
        """Dumps the current algorithm state for convergence tracking."""
        elapsed_time = time.perf_counter_ns() - self.start_time

        log_entry = {
            "iteration": current_iter,
            "wall_clock_time": elapsed_time,
            "nfe_array": self.evaluations_count.copy(),
            "nfe_total": np.sum(self.evaluations_count),
            "front": self.front.copy() if self.front is not None else np.array([]),
            "front_eval": self.front_eval.copy() if self.front_eval is not None else np.array([]),
            "patterns": deepcopy(self.lpatterns),
            "env_version": self.current_env_version,
            "env_params": self.current_env_params
        }

        self.history.append(log_entry)
        self.last_logged_iter = current_iter
        self.lpatterns = []

        if self.verbose:
            logger.info(f"[LOG] Iter: {current_iter:4.0f} | Time: {elapsed_time / 1.e9:6.2f}s | "
                        f"NFE: {log_entry['nfe_array']} | Front size: {len(log_entry['front_eval'])}")

    def save_history(self, filename: str = "agamoo_history.pkl") -> None:
        """Saves the optimization history to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)
        logger.info(f"Convergence history saved to: {filename}")

    async def reevaluate_archive(self, env_version: int = 0, env_params: Dict[str, Any] = {}) -> None:
        """
        Re-evaluates the current Pareto front for new environmental conditions (DMOP)
        and discards dominated solutions.
        """

        self.current_env_version = env_version
        self.current_env_params = env_params

        self._refresh_snapshot_ref()

        if len(self.front) == 0:
            return

        if self.verbose:
            logger.info("Starting asynchronous archive re-evaluation...")

        snapshot_front = self.front.copy()

        futures = []
        target_objs = []
        num_workers = len(self.evaluators)

        # Prepare a new matrix for updated objective function values
        new_front_eval = np.zeros((len(snapshot_front), self.nobjs))

        # Dispatch re-evaluation of old X points for all objectives
        for i in range(self.nobjs):
            if num_workers > 0:
                evaluator = self.evaluators[self.eval_rr_index % num_workers]
                self.eval_rr_index += 1
                futures.append(evaluator.evaluate.remote(snapshot_front, i))
                target_objs.append(i)

        # Asynchronously gather new results
        if futures:
            import asyncio
            results = await asyncio.gather(*futures)
            for idx, res in enumerate(results):
                obj_idx = target_objs[idx]
                new_front_eval[:, obj_idx] = res
                self.evaluations_count[obj_idx] += snapshot_front.shape[0]

        self.total_evaluations = np.min(self.evaluations_count)

        # Re-filtering - remove solutions that became dominated after the environment change
        mask = get_not_dominated(new_front_eval)
        filtered_front = snapshot_front[mask]
        filtered_front_eval = new_front_eval[mask]

        self.front = filtered_front
        self.front_eval = filtered_front_eval

        self._refresh_snapshot_ref()
        if self.verbose:
            logger.info(f"Re-evaluation completed. New archive size: {len(self.front)}")


@ray.remote
class RefHolder:
    """
    Lightweight Ray actor serving as a pointer to the latest global state snapshot.
    This architecture prevents read operations (by Players) from being blocked
    by write operations (in GlobalStorage).
    """
    def __init__(self):
        self.current_ref: Optional[Any] = None

    def update_ref(self, ref_list: List[Any]) -> None:
        """Updates the current snapshot reference. Called by GlobalStorage."""
        self.current_ref = ref_list[0]

    def get_ref(self) -> Any:
        """Retrieves the latest snapshot reference. Called by Players."""
        return self.current_ref


