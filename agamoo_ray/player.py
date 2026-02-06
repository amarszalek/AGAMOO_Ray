import numpy as np
import time
import ray
import traceback
import logging
from abc import ABC, abstractmethod
from agamoo_ray.repair import DefaultRepair
from agamoo_ray.utils import front_suppression
from agamoo_ray.objective import Objective

logger = logging.getLogger(__name__)


class Player(ABC):
    """
    Base class for an autonomous 'Player' entity.
    In the Ray version, concrete implementations of this class (like ClonalSelection)
    should be decorated with @ray.remote to become Actors.
    """

    def __init__(self, num: int, npop: int, objective: Objective, storage_actor,
                 gens='pattern', exchange='mix', verbose=False, init_pop=None):
        self.num = num
        self.npop = npop
        self.objective = objective
        self.storage = storage_actor  # Handle to GlobalStorage Actor
        self.gens = gens
        self.exchange = exchange
        self.verbose = verbose
        self.repair = DefaultRepair()
        self.init_pop = init_pop

        self.ref_holder = None

        self.iteration = 0
        self.evaluation_counter = 0

    def set_repair(self, repair):
        if repair is not None:
            self.repair = repair

    def set_infrastructure(self, storage, ref_holder):
        """Metoda do wstrzykiwania zależności"""
        self.storage = storage
        self.ref_holder = ref_holder

    def start(self):
        """
        Main loop of the player. In Ray, this runs inside the Actor's process.
        """
        if self.verbose:
            logger.info(f"Player {self.num} started (Ray Actor).")

        obj_idx = self.objective.obj
        next_iter_counter = 0
        iters_pop = None

        # Local initialization
        if self.init_pop is not None:
            pop = self.init_pop.copy()
        else:
            pop = self.create_population()

        # Initial Evaluation
        pop = self.repair.do(pop)
        pop_eval = self.objective.evaluate(pop)
        self.evaluation_counter += pop.shape[0]

        if self.verbose:
            logger.info(f"Player {self.num} evaluated init population")

        # Initial update to Global Storage
        # We send the initial population to storage
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
                # Fetch Global State Snapshot
                #global_state = ray.get(self.storage.get_status_flags.remote())
                #global_state = ray.get(self.storage.get_snapshot.remote())

                #snapshot_ref = ray.get(self.storage.get_snapshot_ref.remote())
                #global_state = ray.get(snapshot_ref)

                snapshot_ref = ray.get(self.ref_holder.get_ref.remote())
                if snapshot_ref is None:
                    time.sleep(0.01)
                    continue

                global_state = ray.get(snapshot_ref)

                if global_state['stop_flag']:
                    if self.verbose:
                        logger.info(f"Player {self.num} received stop signal.")
                    break

                patterns = global_state['patterns']
                pattern = patterns[obj_idx]
                next_iter = global_state['next_iter']

                if next_iter <= 0 or next_iter - next_iter_counter > 0:
                    neval = 0

                    if pattern.sum() > 0:
                        try:
                            # Step execution
                            if self.gens == 'all':
                                pop, pop_eval, neval = self.step(pop, pop_eval, np.ones_like(pattern, dtype=bool))
                            else:
                                pop, pop_eval, neval = self.step(pop, pop_eval, pattern)
                        except Exception as e:
                            logger.error(f"Player {self.num} error in step(): {e}", exc_info=True)
                            traceback.print_exc()

                    self.iteration += 1
                    self.evaluation_counter += neval

                    # --- Exchange Logic (Reimplemented with local snapshot) ---
                    #global_state = ray.get(self.storage.get_snapshot.remote())
                    front = global_state['front']
                    front_eval = global_state['front_eval']
                    best = global_state['best']

                    if (self.exchange == 'front_random') and (len(front) > 0):
                        nn = pop.shape[0]
                        inds = np.random.choice(front.shape[0], nn, replace=True)
                        for i in range(nn):
                            # Ensure we don't overwrite the optimized genes (pattern)
                            pop[i, np.logical_not(pattern)] = front[inds[i], np.logical_not(pattern)]

                    elif ('front_sup'in self.exchange) and (len(front) > 0):
                        proc = 100
                        se = self.exchange.split('_')
                        if (len(se) == 3) and (0 < int(se[2]) < 100):
                            proc = int(se[2])

                        # Local shuffle of the front copy
                        arr = np.arange(front.shape[0])
                        np.random.shuffle(arr)
                        local_front = front[arr]
                        local_front_eval = front_eval[arr]

                        target_size = int(pop.shape[0] * (proc / 100))
                        if target_size < local_front.shape[0]:
                            mask = front_suppression(local_front_eval, target_size)
                            local_front = local_front[mask]

                        if len(local_front) > 0:
                            nn = int(pop.shape[0] * (proc / 100))
                            nn = min(nn, local_front.shape[0])
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

                        # Part 1: Mix from Best
                        limit_idx = int(pop.shape[0] * (proc / 100))
                        for i in range(len(best)):
                            if (i != obj_idx) and (best[i] is not None):
                                pop[:limit_idx, patterns[i]] = best[i][patterns[i]]

                        # Part 2: Mix from Front
                        arr = np.arange(front.shape[0])
                        np.random.shuffle(arr)
                        local_front = front[arr]
                        local_front_eval = front_eval[arr]

                        nn = (pop.shape[0] - limit_idx)
                        if nn < local_front.shape[0]:
                            mask = front_suppression(local_front_eval, nn)
                            local_front = local_front[mask]

                        if len(local_front) > 0:
                            # Handle edge case where suppression reduced size too much
                            actual_nn = min(nn, local_front.shape[0])
                            inds = np.random.choice(local_front.shape[0], actual_nn, replace=True)
                            for i in range(actual_nn):
                                pop[limit_idx + i, np.logical_not(pattern)] = local_front[
                                    inds[i], np.logical_not(pattern)]

                    # Final Repair & Evaluate after Exchange
                    pop = self.repair.do(pop)
                    pop_eval = self.objective.evaluate(pop)

                    self.evaluation_counter += pop.shape[0]
                    next_iter_counter += 1

                # 2. Synchronization Check
                # Fetch iteration counters from snapshot
                iters = global_state['iter_counters'].copy()
                iters[obj_idx] = self.iteration  # update local knowledge

                iters_mask = np.zeros(len(iters), dtype=bool)
                for i in range(len(iters)):
                    if iters_pop is None or iters_pop[i] < iters[i]:
                        iters_mask[i] = True

                # 3. Send update to Global Storage
                # If conditions met, send full population, else just heartbeat
                if np.all(iters_mask[:obj_idx]) and np.all(iters_mask[obj_idx + 1:]):
                    # Asynchronous update (Fire and forget)
                    self.storage.update.remote({
                        'nobj': obj_idx,
                        'population': pop.copy(),
                        'population_eval': pop_eval.copy(),
                        'evaluation_counter': self.evaluation_counter,  # diff since last update
                        'iteration': self.iteration,
                        'iter_flag': False
                    })
                    if self.verbose:
                        logger.info(f"Player {self.num} sent pop at iter {self.iteration}")

                    next_iter_counter = 0
                    iters_pop = iters.copy()
                else:
                    # Heartbeat update (only iteration info)
                    self.storage.update.remote({
                        'nobj': obj_idx,
                        'iter_flag': True,
                        'iteration': self.iteration
                    })
                    # Small sleep to prevent busy loop hammering the storage if waiting
                    time.sleep(0.001)

        except Exception as e:
            logger.error(f"Player {self.num} crashed: {e}", exc_info=True)
            traceback.print_exc()
        finally:
            if self.verbose:
                logger.info(f"Player {self.num} exiting.")

    @abstractmethod
    def step(self, pop, pop_eval, pattern):
        """
        Abstract method to implement in subclasses.
        This method will be called repeatedly in the loop.
        """
        raise NotImplementedError('You must override this method in your class!')

    def evaluate_only(self, pop):
        """Metoda pomocnicza dla GlobalStorage do doliczania brakujących kryteriów."""
        return self.objective.evaluate(pop)

    def create_population(self):
        pop = np.zeros((self.npop, self.objective.n_var))
        for i in range(self.npop):
            pop[i] = self._create_individual_uniform(self.objective.bounds)
        return pop

    @staticmethod
    def _create_individual_uniform(bounds):
        a = np.array([bounds[k][0] for k in range(len(bounds))])
        b = np.array([bounds[k][1] for k in range(len(bounds))])
        return np.random.uniform(a, b)


@ray.remote
class Evaluator:
    """
    Dedykowany aktor do przeliczania funkcji celu.
    Działa jako 'kalkulator' dla GlobalStorage.
    """
    def __init__(self, objectives):
        self.objectives = objectives

    def evaluate(self, pop, i):
        res = self.objectives[i].evaluate(pop)
        return np.array(res).flatten()