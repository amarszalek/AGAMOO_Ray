import numpy as np
import ray
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional

from agamoo_ray.player import Player
from agamoo_ray.objective import Objective


@ray.remote
class SimulatedAnnealing(Player):
    """
        Asynchronous Ray Actor implementing the Simulated Annealing (SA) Algorithm.
        W architekturze populacyjnej każdy osobnik stanowi niezależny łańcuch wyżarzania,
        dzieląc wspólną temperaturę globalną roju.
    """

    def __init__(self,
                 num: int,
                 npop: int,
                 player_param: Dict[str, Any],
                 objective: Objective,
                 storage_actor: Any,
                 gens: str = 'pattern',
                 exchange: str = 'front_sup',
                 verbose: bool = False,
                 init_pop: Optional[np.ndarray] = None):
        """
        Initializes the Simulated Annealing Player.

        Args:
            num (int): Unique identifier index for the player.
            npop (int): Population size (liczba równoległych łańcuchów SA).
            player_param (Dict[str, Any]): Hyperparameters for the SA algorithm:
                - 'T0': Initial temperature (Temperatura początkowa).
                - 'T_min': Minimum temperature (Temperatura minimalna).
                - 'step_size': Wielkość kroku perturbacji jako ułamek domeny (np. 0.05 to 5%).
                - 'max_eval': Max number of evaluations (instead of cooling_rate)
                - 'create' (str): Create population method ('uniform', 'lhs')
            objective (Objective): The objective function to optimize.
            storage_actor (Any): Handle to the GlobalStorage Ray Actor.
            gens (str): Gene allocation strategy ('pattern' or 'all').
            exchange (str): Gene exchange strategy for cooperative coevolution.
            verbose (bool): Enables detailed execution logging.
            init_pop (np.ndarray, optional): Custom initial population array.
        """

        self.T0: float = player_param.get('T0', 100.0)
        self.T_min: float = player_param.get('T_min', 1e-5)
        self.step_size: float = player_param.get('step_size', 0.05)
        self.max_eval: int = player_param.get('max_eval', 10000)
        self.create: str = player_param.get('create', 'lhs')
        self.seed = player_param.get('seed', None)
        self.dim = objective.n_var

        if self.seed is not None:
            np.random.seed(self.seed + num)

        super().__init__(num, npop, objective, storage_actor, gens, exchange, verbose, init_pop, create_method=self.create)

        # Inicjalizacja aktualnej temperatury
        self.T: float = self.T0

    def step(self, pop: np.ndarray, pop_eval: np.ndarray, pattern: np.ndarray,
             global_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Executes a single evolutionary cycle of the Simulated Annealing algorithm.

        Args:
            pop (np.ndarray): Current population.
            pop_eval (np.ndarray): Evaluated objective values.
            pattern (np.ndarray): Boolean mask indicating modifiable decision variables.
            global_state: Dictionary containing global optimization state.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Updated population, updated evaluations, and number of evaluations.
        """
        evaluation_counter: int = 0
        n_pop = pop.shape[0]

        # Dynamiczne obliczanie Temperatury na podstawie max_eval
        if (global_state is not None) and ('evaluations_count' in global_state):
            # Pobieramy faktyczną liczbę ewaluacji dla tego gracza
            current_evals = global_state['evaluations_count'][self.objective.obj]
            # Ułamek postępu: 0.0 (start) do 1.0 (koniec)
            progress = min(current_evals / max(1, self.max_eval), 1.0)

            # Wzór na chłodzenie wykładnicze idealnie rozciągnięte w czasie:
            # T = T0 * (T_min / T0)^progress
            self.T = self.T0 * ((self.T_min / self.T0) ** progress)

        # Pobranie granic problemu
        bounds_arr = np.array(self.objective.bounds)
        a = bounds_arr[:, 0]
        b = bounds_arr[:, 1]

        # Obliczenie rozstępu między granicami (dla dostosowania wielkości perturbacji do skali problemu)
        domain_range = b - a

        temp_pop = deepcopy(pop)
        temp_pop_eval = deepcopy(pop_eval)

        # Generowanie sąsiadów (Perturbacja)
        # Generujemy macierz szumu o rozkładzie normalnym i skalujemy go parametrem step_size i domeną
        noise = np.random.randn(n_pop, self.dim) * self.step_size * domain_range
        new_pop_all = temp_pop + noise

        # Aplikujemy wzorzec genów (Zmieniamy pozycje tylko dla przypisanych przez DVA genów)
        new_pop = np.where(pattern, new_pop_all, temp_pop)

        # Zabezpieczenie ograniczeń przestrzeni
        new_pop = np.clip(new_pop, a, b)

        # Naprawa i Ewaluacja (Pełna Wektoryzacja)
        new_pop = self.repair.do(new_pop)
        new_pop_eval = self.objective.evaluate(new_pop).flatten()
        evaluation_counter += n_pop

        # Selekcja i Prawdopodobieństwo Boltzmanna
        delta_f = new_pop_eval - temp_pop_eval

        # Rozwiązania lepsze (delta_f < 0) akceptujemy zawsze
        better_mask = delta_f < 0

        # Rozwiązania gorsze akceptujemy z prawdopodobieństwem exp(-delta_f / T)
        prob = np.zeros(n_pop)
        worse_mask = delta_f >= 0

        # Zabezpieczenie przed underflow (bardzo ujemne wartości exponenty dają po prostu 0.0 w prawdopodobieństwie)
        exponent = np.clip(-delta_f[worse_mask] / self.T, -700, 0)
        prob[worse_mask] = np.exp(exponent)

        random_vals = np.random.rand(n_pop)
        accept_worse_mask = worse_mask & (random_vals < prob)

        # Połączenie obu masek (lepsze OR gorsze, ale zaakceptowane)
        accept_mask = better_mask | accept_worse_mask

        # Aktualizacja zaakceptowanych rozwiązań
        temp_pop[accept_mask] = new_pop[accept_mask]
        temp_pop_eval[accept_mask] = new_pop_eval[accept_mask]

        return temp_pop, temp_pop_eval, evaluation_counter

