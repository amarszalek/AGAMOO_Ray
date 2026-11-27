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
                - 'create' (str): Create population method ('uniform', 'lhs')
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
        self.create: str = player_param.get('create', 'lhs')
        self.seed = player_param.get('seed', None)
        self.dim = objective.n_var
        self.guide: str = player_param.get('guide', 'global')  # 'global' | 'local'
        self.k_guide: int = player_param.get('k_guide', 5)
        # What happens to particles replaced by an exchange:
        #   'none'   - nothing (pbest and velocity kept),
        #   'keep_v' - pbest := new (archive) position, velocity kept; safe with guide='local',
        #              where a particle may be its own guide and would freeze with zero velocity,
        #   'zero_v' - pbest := new position and velocity := 0 (full restart of the particle).
        # bool values are accepted for backward compatibility: True -> 'keep_v', False -> 'none'.
        self.reset_pbest: str = self._parse_reset_pbest(player_param.get('reset_pbest', 'none'))
        # Mutacja wielomianowa (Deb) po ruchu roju, tylko na zmiennych z pattern.
        #   mutation_rate - odsetek cząstek mutowanych w kroku (0.0 = wyłączona, zachowanie jak dotąd),
        #   eta_m         - indeks rozkładu (większy = mniejsze skoki); w wybranej cząstce każda
        #                   zmienna z pattern mutuje z prawdop. 1/|pattern| (co najmniej jedna).
        # Prędkości i pbest bez zmian: pbest przyjmie skok tylko przy poprawie.
        self.mutation_rate: float = float(player_param.get('mutation_rate', 0.0))
        self.eta_m: float = float(player_param.get('eta_m', 20.0))
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"PSO: mutation_rate must be in [0, 1], got {self.mutation_rate}")
        if self.eta_m < 0.0:
            raise ValueError(f"PSO: eta_m must be >= 0, got {self.eta_m}")

        if self.seed is not None:
            np.random.seed(self.seed + num)

        super().__init__(num, npop, objective, storage_actor, gens, exchange, verbose, init_pop, create_method=self.create)

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
            front_obj = global_state['front_eval'][:, self.objective.obj]
            if self.guide == 'local' and len(front) > 1:
                span = np.where(b - a > 0, b - a, 1.0)
                P, Q = pop / span, front / span
                d2 = (P ** 2).sum(1)[:, None] + (Q ** 2).sum(1)[None, :] - 2.0 * P @ Q.T
                k = min(self.k_guide, front.shape[0])
                nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
                gbest_pos = front[nn[np.arange(n_pop), np.argmin(front_obj[nn], axis=1)]]  # (n_pop, dim)
            else:
                gbest_pos = front[np.argmin(front_obj)].copy()
        else:
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

        if self.mutation_rate > 0.0:
            new_pop = self._polynomial_mutation(new_pop, pattern, a, b)

        # Naprawa i Ewaluacja (Batching)
        #new_pop = self.repair.do(new_pop)
        if hasattr(self.repair, 'order'):
            new_pop = self.repair.feasible(new_pop)
            idx = self.repair.order(new_pop)
            new_pop = np.take_along_axis(new_pop, idx, axis=1)
            new_velocities = np.take_along_axis(new_velocities, idx, axis=1)
            self.velocities = np.take_along_axis(self.velocities, idx, axis=1)
        else:
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

    def on_environment_change(self, pop: np.ndarray, pop_eval: np.ndarray) -> None:
        # pbest_eval belongs to the previous environment; restart personal memory from the
        # freshly re-evaluated population. Velocities are kept (they carry no fitness info).
        if self.pbest_pos is not None:
            self.pbest_pos = pop.copy()
            self.pbest_eval = pop_eval.copy()

    def _polynomial_mutation(self, pop: np.ndarray, pattern: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(pattern)
        if idx.size == 0:
            return pop
        rows = np.flatnonzero(np.random.rand(pop.shape[0]) < self.mutation_rate)
        if rows.size == 0:
            return pop
        mask = np.random.rand(rows.size, idx.size) < 1.0 / idx.size
        none = ~mask.any(axis=1)
        mask[none, np.random.randint(idx.size, size=int(none.sum()))] = True   # co najmniej jedna zmienna

        sub = pop[np.ix_(rows, idx)]
        lo, span = a[idx], b[idx] - a[idx]
        u = np.random.rand(*sub.shape)
        e = 1.0 / (self.eta_m + 1.0)
        delta = np.where(u < 0.5, (2.0 * u) ** e - 1.0, 1.0 - (2.0 * (1.0 - u)) ** e)
        sub = np.where(mask, np.clip(sub + delta * span, lo, lo + span), sub)

        out = pop.copy()
        out[np.ix_(rows, idx)] = sub
        return out

    @staticmethod
    def _parse_reset_pbest(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return 'keep_v' if value else 'none'
        value = str(value)
        if value not in ('none', 'keep_v', 'zero_v'):
            raise ValueError(f"PSO: reset_pbest must be 'none', 'keep_v' or 'zero_v', got {value!r}")
        return value

    def on_exchange(self, pop: np.ndarray, pop_eval: np.ndarray, modified_mask: np.ndarray) -> None:
        if self.reset_pbest == 'none' or self.pbest_pos is None:
            return  # disabled, or swarm not initialised yet (first step builds pbest from pop)
        self.pbest_pos[modified_mask] = pop[modified_mask]
        self.pbest_eval[modified_mask] = pop_eval[modified_mask]
        if self.reset_pbest == 'zero_v':
            self.velocities[modified_mask] = 0.0

