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

    def _levy_flight(self, n: int = 1):
        """
        Zwektoryzowana wersja lotów Lévy'ego.
        Zwraca macierz o wymiarach (n, dim).
        """
        num = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        den = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        sigma = (num / den) ** (1 / self.beta)

        u = np.random.normal(0, sigma, (n, self.dim))
        v = np.random.normal(0, 1, (n, self.dim))

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
        a = bounds_arr[:, 0]
        b = bounds_arr[:, 1]
        domain_range = b - a

        n_pop = pop.shape[0]

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

        # Generujemy krok dla całej populacji na raz
        dynamic_alpha = self.alpha * domain_range
        step_size = dynamic_alpha * self._levy_flight(n_pop) * (temp_pop - best_nest)
        new_nests_all = temp_pop + step_size * np.random.randn(n_pop, self.dim)

        # Aplikujemy wzorzec genów (broadcasting zadziała automatycznie)
        new_nests = np.where(pattern, new_nests_all, temp_pop)
        new_nests = np.clip(new_nests, a, b)

        # Naprawa i ocena całej populacji w jednym wywołaniu (batching)
        new_nests = self.repair.do(new_nests)
        new_nests_eval = self.objective.evaluate(new_nests).flatten()  # Upewniamy się, że to wektor 1D
        evaluation_counter += n_pop

        # Zastępujemy tylko te gniazda, które są lepsze (maska logiczna NumPy)
        better_mask = new_nests_eval < temp_pop_eval
        temp_pop[better_mask] = new_nests[better_mask]
        temp_pop_eval[better_mask] = new_nests_eval[better_mask]

        # Wyznaczamy, które gniazda zostały odkryte przez gospodarza
        abandon_mask = np.random.rand(n_pop) < self.pa
        num_abandoned = np.sum(abandon_mask)

        if num_abandoned > 0:
            # Wybieramy losowe pary gniazd do wyznaczenia kierunku skoku
            idx1 = np.random.randint(0, n_pop, num_abandoned)
            idx2 = np.random.randint(0, n_pop, num_abandoned)

            # Wektoryzacja skoku (używamy reshape(num_abandoned, 1) do prawidłowego mnożenia macierzy)
            random_steps = np.random.rand(num_abandoned, 1) * (temp_pop[idx1] - temp_pop[idx2])
            new_nests_all_ab = temp_pop[abandon_mask] + random_steps

            new_nests_ab = np.where(pattern, new_nests_all_ab, temp_pop[abandon_mask])
            new_nests_ab = np.clip(new_nests_ab, a, b)

            # Naprawa i ocena porzuconych gniazd
            new_nests_ab = self.repair.do(new_nests_ab)
            new_nests_ab_eval = self.objective.evaluate(new_nests_ab).flatten()
            evaluation_counter += num_abandoned

            if self.strategy == 'algorithm':
                # Zachłanne zastąpienie (tylko lepsze rozwiązania)
                current_eval_ab = temp_pop_eval[abandon_mask]
                better_mask_ab = new_nests_ab_eval < current_eval_ab

                # Aktualizacja oryginalnych macierzy za pomocą masek
                temp_pop_abandoned = temp_pop[abandon_mask]
                temp_pop_abandoned[better_mask_ab] = new_nests_ab[better_mask_ab]
                temp_pop[abandon_mask] = temp_pop_abandoned

                temp_pop_eval_abandoned = temp_pop_eval[abandon_mask]
                temp_pop_eval_abandoned[better_mask_ab] = new_nests_ab_eval[better_mask_ab]
                temp_pop_eval[abandon_mask] = temp_pop_eval_abandoned

            else:
                # 'nature': Bezwarunkowe zastąpienie gniazda
                temp_pop[abandon_mask] = new_nests_ab
                temp_pop_eval[abandon_mask] = new_nests_ab_eval

        return temp_pop, temp_pop_eval, evaluation_counter

