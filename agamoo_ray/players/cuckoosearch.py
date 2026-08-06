import numpy as np
import ray
import math
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional, List

from agamoo_ray.player import Player
from agamoo_ray.objective import Objective


@ray.remote
class CuckooSearch(Player):
    """
        Asynchronous Ray Actor implementing the Cuckoo Search Algorithm.
    """

    def __init__(self,
                 num: int,
                 npop: int,
                 player_param: Dict[str, Any],
                 objective: Objective,
                 storage_actor: Any,
                 gens: str = 'pattern',
                 exchange: str ='front_sup',
                 verbose: bool = False,
                 init_pop: Optional[np.ndarray] = None):
        """
        Initializes the Cuckoo Search.

        Args:
            num (int): Unique identifier index for the player.
            npop (int): Population size (number of antibodies).
            player_param (Dict[str, Any]): Hyperparameters for the CSA algorithm:
                - 'pa': Prawdopodobieństwo odkrycia kukułczego jaja przez gospodarza.
                - 'beta': Parametr rozkładu Lévy'ego.
                - 'alpha': Mnożnik skoku (współczynnik skali).
                - 'bounds': Krotka z ograniczeniami (min_bound, max_bound).
                - 'strategy' (str): Selection strategy ('algorithm' or 'nature')
            objective (Objective): The objective function to optimize.
            storage_actor (Any): Handle to the GlobalStorage Ray Actor.
            gens (str): Gene allocation strategy ('pattern' or 'all').
            exchange (str): Gene exchange strategy for cooperative coevolution.
            verbose (bool): Enables detailed execution logging.
            init_pop (np.ndarray, optional): Custom initial population array.
        """
        self.pa: float = player_param.get('pa', 0.25)
        self.beta: float = player_param.get('beta', 1.5)
        self.alpha: float = player_param.get('alpha', 0.01)
        self.max_eval: int = player_param.get('max_eval', 10000)
        self.strategy: str = player_param.get('strategy', 'algorithm')
        self.seed = player_param.get('seed', None)
        self.dim = objective.n_var
        if self.seed is not None:
            np.random.seed(self.seed + num)

        super().__init__(num, npop, objective, storage_actor, gens, exchange, verbose, init_pop)

    def _levy_flight(self):
        """
        Generuje krok z rozkładu Lévy'ego przy użyciu algorytmu Mantegny.
        Formuła: s = u / |v|^(1/beta)
        """
        num = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        den = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        sigma = (num / den) ** (1 / self.beta)

        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)

        step = u / (np.abs(v) ** (1 / self.beta))
        return step

    def step(self, pop: np.ndarray, pop_eval: np.ndarray, pattern: np.ndarray, global_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Executes a single evolutionary cycle of the Clonal Selection algorithm.

        Args:
            pop (np.ndarray): Current population (antibodies).
            pop_eval (np.ndarray): Evaluated objective values (affinity).
            pattern (np.ndarray): Boolean mask indicating modifiable decision variables.
            global_state: new

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Updated population, updated evaluations, and number of evaluations.
        """

        evaluation_counter: int = 0
        bounds_arr = np.array(self.objective.bounds)
        a = bounds_arr[pattern, 0]
        b = bounds_arr[pattern, 1]

        if global_state is not None and len(global_state.get('front', [])) > 1:
            front = global_state['front']
            front_eval = global_state['front_eval'][:,self.objective.obj]
            best_nest = front[np.argmin(front_eval)].copy()
            best_fitness = np.min(front_eval)
        else:
            best_nest = pop[np.argmin(pop_eval)].copy()
            best_fitness = np.min(pop_eval)

        temp_pop = deepcopy(pop)
        temp_pop_eval = deepcopy(pop_eval)

        for i in range(pop.shape[0]):
            step_size = self.alpha * self._levy_flight() * (temp_pop[i] - best_nest)
            new_nest = temp_pop[i] + step_size * np.random.randn(self.dim)
            new_nest = np.clip(new_nest, a, b)

            new_nest = self.repair.do(new_nest.reshape((1,-1)))
            new_nest_eval = self.objective.evaluate(new_nest)
            evaluation_counter += new_nest.shape[0]

            # Jeśli nowe gniazdo jest lepsze, zastąp je
            if new_nest_eval[0] < temp_pop_eval[i]:
                temp_pop[i] = new_nest[0]
                temp_pop_eval[i] = new_nest_eval[0]

                # Aktualizacja najlepszego rozwiązania globalnego
                if new_nest_eval[0] < best_fitness:
                    best_fitness = new_nest_eval[0]
                    best_nest = new_nest[0].copy()

        # 2. Odkrywanie gniazd (z prawdopodobieństwem pa) i budowanie nowych
        for i in range(temp_pop.shape[0]):
            if np.random.rand() < self.pa:
                step_size = np.random.rand() * (temp_pop[np.random.randint(0, temp_pop.shape[0])] -
                                                temp_pop[np.random.randint(0, temp_pop.shape[0])])
                new_nest = temp_pop[i] + step_size
                new_nest = np.clip(new_nest, a, b)

                new_nest = self.repair.do(new_nest.reshape((1, -1)))
                new_nest_eval = self.objective.evaluate(new_nest)
                evaluation_counter += new_nest.shape[0]

                if self.strategy == 'algorithm':
                    if new_nest_eval[0] < temp_pop_eval[i]:
                        temp_pop[i] = new_nest[0]
                        temp_pop_eval[i] = new_nest_eval[0]
                else:
                    temp_pop[i] = new_nest[0]
                    temp_pop_eval[i] = new_nest_eval[0]

                if new_nest_eval[0] < best_fitness:
                    best_fitness = new_nest_eval[0]
                    best_nest = new_nest[0].copy()

        return temp_pop, temp_pop_eval, evaluation_counter

