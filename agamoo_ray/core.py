import time
import ray
import numpy as np
import logging
import asyncio
import traceback
import pickle
from copy import deepcopy
from tqdm.auto import tqdm
from agamoo_ray.utils import get_not_dominated, front_suppression, assigning_gens, adaptive_linear_assigning_gens
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
                 next_iter: int,
                 max_front: int,
                 max_front_tol: float = 0.0,
                 init_pop: str = 'separate',
                 assign_gens: str = 'random',
                 front_f: Optional[Callable] = None,
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
        self.next_iter = next_iter
        self.max_front = max_front
        self.max_front_tol = max_front_tol
        self.init_pop = init_pop
        self.front_f = front_f
        self.verbose = verbose
        self.assign_gens = assign_gens
        self.log_freq = log_freq

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
            nvars, nobjs, self.max_eval, self.change_iter, self.next_iter,
            self.max_front, self.assign_gens, self.max_front_tol, self.front_f,
            ref_holder=self.ref_holder, verbose=self.verbose, log_freq=self.log_freq,
        )
        return self.storage

    def start_optimize(self, tqdm_disable: bool = False) -> None:
        """
        Starts the asynchronous optimization process and monitors the stop conditions.

        Args:
            tqdm_disable (bool): If True, disables the progress bar.
        """
        if not self.players:
            raise ValueError("Players list is empty. Initialize players first.")
        if not self.storage:
            raise ValueError("Storage not created. Call create_storage() first.")

        # Reset global state for a fresh run
        ray.get(self.storage.reset.remote())

        if self.verbose:
            logger.info("Starting AGAMOO optimization using Ray...")

        # Start asynchronous player loops
        for p in self.players:
            p.set_repair.remote(self.repair)
            p.start.remote()

        if self.verbose:
            logger.info("Players are running in the background...")

        # Driver loop: Monitoring the global state via non-blocking RefHolder
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

        if self.verbose:
            logger.info("Optimization finished.")

        # Terminate player actors
        for p in self.players:
            ray.kill(p)

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
                 next_iter: int,
                 max_front: int,
                 assign_gens: str = 'random',
                 max_front_tol: float = 0.0,
                 front_f: Optional[Callable] = None,
                 verbose: bool = False,
                 ref_holder: Optional[Any] = None,
                 log_freq=0):

        self.nvars = nvars
        self.nobjs = nobjs
        self.max_eval = max_eval
        self.change_iter = change_iter
        self.next_iter = next_iter
        self.max_front = max_front
        self.max_front_tol = max_front_tol
        self.front_f = front_f

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
            'stop_flag': self.stop_flag,
            'evaluations': self.total_evaluations
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

    def get_status(self) -> Dict[str, Any]:
        """Returns the current optimization status."""
        return {
            'iterations': self.iter_counters,
            'evaluations': self.total_evaluations,
            'stop_flag': self.stop_flag,
            'front_size': len(self.front)
        }

    def get_results(self) -> Dict[str, Any]:
        """Returns the final, suppressed Pareto front and evaluations stats."""
        final_front = self.front
        final_front_eval = self.front_eval

        if len(final_front) > self.max_front:
            mask = front_suppression(final_front_eval, self.max_front)
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

    async def update(self, data: Dict[str, Any]) -> None:
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

            # Dynamic Variable Assignment logic
            min_iter = np.min(self.iter_counters)
            if min_iter - self.min_iter_pop >= self.change_iter:
                if self.assign_gens=='random':
                    self.patterns = assigning_gens(self.nvars, self.nobjs)
                elif self.assign_gens=='adaptive_linear':
                    self.patterns = adaptive_linear_assigning_gens(self.front, self.front_eval, self.nvars, self.nobjs)
                else:
                    raise ValueError(f"Unknown assign_gens strategy: {self.assign_gens}")
                self.min_iter_pop = min_iter

            self.lpatterns.append(self.patterns.copy())

            # Return early if it's just a heartbeat
            if data.get('iter_flag', False):
                return

            # Extract population data
            pop = data['population']
            pop_eval_partial = data['population_eval']
            neval = data['evaluation_counter']

            if neval > 0:
                self.evaluations_count[nobj] = neval
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
                mask = front_suppression(self.front_eval, self.max_front)
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
            "front_eval": self.front_eval.copy() if self.front_eval is not None else np.array([]),
            "patterns": deepcopy(self.lpatterns)
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


