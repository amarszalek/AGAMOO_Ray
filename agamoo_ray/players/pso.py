import numpy as np
import ray
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional

from agamoo_ray.player import Player
from agamoo_ray.objective import Objective


@ray.remote
class PSO(Player):
    """
        Asynchronous Ray Actor implementing the Particle Swarm Optimization (PSO) Algorithm.
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
        Initializes the Particle Swarm Optimization Player.

        Args:
            num (int): Unique identifier index for the player.
            npop (int): Population size (number of particles).
            player_param (Dict[str, Any]): Hyperparameters for the PSO algorithm:
                - 'w': Inertia weight (Współczynnik bezwładności).
                - 'c1': Cognitive parameter (Współczynnik uczenia lokalnego - dążenie do pbest).
                - 'c2': Social parameter (Współczynnik uczenia globalnego - dążenie do gbest).
            objective (Objective): The objective function to optimize.
            storage_actor (Any): Handle to the GlobalStorage Ray Actor.
            gens (str): Gene allocation strategy ('pattern' or 'all').
            exchange (str): Gene exchange strategy for cooperative coevolution.
            verbose (bool): Enables detailed execution logging.
            init_pop (np.ndarray, optional): Custom initial population array.
        """
        self.w: float = player_param.get('w', 0.729)
        self.c1: float = player_param.get('c1', 1.49445)
        self.c2: float = player_param.get('c2', 1.49445)
        self.seed = player_param.get('seed', None)
        self.dim = objective.n_var

        if self.seed is not None:
            np.random.seed(self.seed + num)

        super().__init__(num, npop, objective, storage_actor, gens, exchange, verbose, init_pop)

        # Wewnętrzny stan roju (inicjalizowany przy pierwszym kroku)
        self.velocities: Optional[np.ndarray] = None
        self.pbest_pos: Optional[np.ndarray] = None
        self.pbest_eval: Optional[np.ndarray] = None


    def step(self, pop: np.ndarray, pop_eval: np.ndarray, pattern: np.ndarray, global_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Executes a single evolutionary cycle of the Particle Swarm Optimization algorithm.

        Args:
            pop (np.ndarray): Current population (positions of particles).
            pop_eval (np.ndarray): Evaluated objective values.
            pattern (np.ndarray): Boolean mask indicating modifiable decision variables.
            global_state: Dictionary containing global optimization state (e.g., Pareto front).

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Updated population, updated evaluations, and number of evaluations.
        """
        evaluation_counter: int = 0
        n_pop = pop.shape[0]

        bounds_arr = np.array(self.objective.bounds)
        a = bounds_arr[:, 0]
        b = bounds_arr[:, 1]

        # Inicjalizacja stanu roju (tylko w pierwszej iteracji)
        if self.velocities is None:
            # Prędkość początkowa losowana w małym przedziale, np. od -10% do 10% rozpiętości domeny
            v_max = (b - a) * 0.1
            self.velocities = np.random.uniform(-v_max, v_max, (n_pop, self.dim))
            self.pbest_pos = deepcopy(pop)
            self.pbest_eval = deepcopy(pop_eval)
        else:
            # Zabezpieczenie przed wpływem Koewolucji (Cooperative Coevolution).
            # Jeśli inny algorytm zmodyfikował naszą populację i jest ona lepsza niż nasz dotychczasowy pbest, aktualizujemy go.
            better_mask_ext = pop_eval < self.pbest_eval
            self.pbest_pos[better_mask_ext] = pop[better_mask_ext]
            self.pbest_eval[better_mask_ext] = pop_eval[better_mask_ext]

        # Ustalenie Global Best (gbest)
        if global_state is not None and len(global_state.get('front', [])) > 0:
            front = global_state['front']
            front_eval = global_state['front_eval'][:, self.objective.obj]
            gbest_pos = front[np.argmin(front_eval)].copy()
        else:
            # Fallback, jeśli front globalny jeszcze nie istnieje
            gbest_pos = self.pbest_pos[np.argmin(self.pbest_eval)].copy()

        # Aktualizacja prędkości i pozycji (Pełna wektoryzacja)
        # Losowe macierze r1 i r2 (unikalne dla każdej cząstki i każdego wymiaru)
        r1 = np.random.rand(n_pop, self.dim)
        r2 = np.random.rand(n_pop, self.dim)

        # Wektorowe obliczenie nowej prędkości (Inertia + Cognitive + Social)
        cognitive = self.c1 * r1 * (self.pbest_pos - pop)
        social = self.c2 * r2 * (gbest_pos - pop)
        new_velocities = self.w * self.velocities + cognitive + social

        # Obliczenie nowej pozycji
        new_pop_all = pop + new_velocities

        # Zastosowanie maski (Zmieniamy pozycje tylko dla przypisanych przez DVA genów)
        new_pop = np.where(pattern, new_pop_all, pop)

        # Zabezpieczenie ograniczeń przestrzeni
        new_pop = np.clip(new_pop, a, b)

        # Naprawa i Ewaluacja (Batching)
        new_pop = self.repair.do(new_pop)
        new_pop_eval = self.objective.evaluate(new_pop).flatten()
        evaluation_counter += n_pop

        # Aktualizacja stanu wewnętrznego
        # Aktualizacja Personal Best (pbest)
        better_mask = new_pop_eval < self.pbest_eval
        self.pbest_pos[better_mask] = new_pop[better_mask]
        self.pbest_eval[better_mask] = new_pop_eval[better_mask]

        # Nadpisanie wektora prędkości. Aktualizujemy tylko tam, gdzie działa pattern,
        # aby uśpione geny nie kumulowały w tle "ukrytej" energii kinetycznej.
        self.velocities = np.where(pattern, new_velocities, self.velocities)

        return new_pop, new_pop_eval, evaluation_counter

