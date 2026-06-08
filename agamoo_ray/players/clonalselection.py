import numpy as np
import ray
from copy import deepcopy
from typing import Dict, Any, Tuple, Optional, List

from agamoo_ray.player import Player
from agamoo_ray.objective import Objective


@ray.remote
class ClonalSelection(Player):
    """
        Asynchronous Ray Actor implementing the Clonal Selection Algorithm (CSA).

        Inspired by the biological immune system's principle of clonal selection,
        this actor performs local and global search operations (cloning, hypermutation,
        and selection based on affinity) on a specific subset of decision variables.
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
        Initializes the Clonal Selection Player.

        Args:
            num (int): Unique identifier index for the player.
            npop (int): Population size (number of antibodies).
            player_param (Dict[str, Any]): Hyperparameters for the CSA algorithm:
                - 'nclone' (int): Maximum number of clones generated for the best individual.
                - 'mutate_args' (List[float]): Probabilities/parameters for [uniform, gaussian, bound] mutations.
                - 'sup' (float): Suppression factor (0.0 to 1.0) for diversity injection.
                - 'strategy' (str): Selection strategy ('base' or 'all_best').
            objective (Objective): The objective function (antigen) to optimize.
            storage_actor (Any): Handle to the GlobalStorage Ray Actor.
            gens (str): Gene allocation strategy ('pattern' or 'all').
            exchange (str): Gene exchange strategy for cooperative coevolution.
            verbose (bool): Enables detailed execution logging.
            init_pop (np.ndarray, optional): Custom initial population array.
        """
        # Extract CSA-specific hyperparameters
        self.nclone: int = player_param.get('nclone', 15)
        self.mutate_args: Tuple[float, ...] = tuple(player_param.get('mutate_args', [0.45, 0.9, 0.01]))
        self.sup: float = player_param.get('sup', 0.0)
        self.strategy: str = player_param.get('strategy', 'base')
        self.scalar_freq: int = player_param.get('scalar_freq', 0)
        self.max_eval: int = player_param.get('max_eval', 10000)
        self.theta: float = player_param.get('theta', 5.0)
        self.persistent_w = None
        self.seed  = player_param.get('seed', None)
        if self.seed is not None:
            np.random.seed(self.seed+num)

        # Initialize the base Player class
        super().__init__(num, npop, objective, storage_actor, gens, exchange, verbose, init_pop)

    def step(self, pop: np.ndarray, pop_eval: np.ndarray, pattern: np.ndarray, global_state: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Executes a single evolutionary cycle of the Clonal Selection algorithm.

        Steps:
        1. Affinity Evaluation (Sorting based on objective values).
        2. Proliferation (Cloning inversely proportional to rank).
        3. Affinity Maturation (Hypermutation of clones).
        4. Selection (Replacing parent if clone is superior).
        5. Receptor Editing / Suppression (Injecting random diversity).

        Args:
            pop (np.ndarray): Current population (antibodies).
            pop_eval (np.ndarray): Evaluated objective values (affinity).
            pattern (np.ndarray): Boolean mask indicating modifiable decision variables.
            global_state: new

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: Updated population, updated evaluations, and number of evaluations.
        """
        temp_pop = deepcopy(pop)
        temp_pop_eval = deepcopy(pop_eval)

        # --- DYNAMICZNA SKALARYZACJA (Surogatowa) ---
        use_scalarization = False

        # BEZPIECZNA ITERACJA: Pobieramy aktualną iterację z globalnego licznika dla tego gracza
        current_iter = 0
        if global_state is not None and 'iter_counters' in global_state:
            # self.objective.obj to indeks przypisany do tego gracza (np. 0, 1, 2...)
            current_iter = int(global_state['iter_counters'][self.objective.obj])

        if global_state is not None and len(global_state.get('front', [])) > 5 and self.scalar_freq > 0 :
            # Wprowadzamy cykle: np. 5 iteracji specjalizacji + 5 iteracji kompromisu
            cycle_length = self.scalar_freq
            phase_step = current_iter % cycle_length

            if phase_step >= 0:#self.scalar_freq:
                use_scalarization = True
                front = global_state['front']
                front_eval = global_state['front_eval']
                #_, unique_indices = np.unique(front_eval, axis=0, return_index=True)
                #front = front[unique_indices]
                #front_eval = front_eval[unique_indices]

                current_evals = global_state.get('evaluations', 0)
                progress = min(current_evals / max(1, self.max_eval), 1.0)

                # GENEROWANIE WEKTORA TYLKO NA POCZĄTKU FAZY (lub gdy go brak)
                if phase_step == 0 or self.persistent_w is None:
                    w_focused = np.full(front_eval.shape[1], 0.01)
                    w_focused[self.objective.obj] = 1.0
                    w_random = np.random.uniform(0.01, 1.0, front_eval.shape[1])

                    w_raw = (1.0 - progress) * w_focused + (progress * w_random)
                    self.persistent_w = w_raw / np.linalg.norm(w_raw)

                # Algorytm przez całą fazę (np. przez 5 iteracji) używa zapamiętanego wektora!
                w = self.persistent_w

                # Obliczenie punktu idealnego i nadiru z globalnego frontu
                ideal = np.min(front_eval, axis=0)
                nadir = np.max(front_eval, axis=0)
                denom = nadir - ideal
                denom[denom < 1e-9] = 1e-9


                theta = self.theta * (progress ** 2)
                #theta = self.theta

                # Funkcja pomocnicza: Estymacja sąsiada + Obliczenie PBI
                def calc_pbi(x_array, exact_evals):
                    # Szukanie najbliższego sąsiada w przestrzeni X
                    diff = x_array[:, np.newaxis, :] - front[np.newaxis, :, :]
                    dist_sq = np.sum(diff ** 2, axis=2)
                    nearest_idx = np.argmin(dist_sq, axis=1)

                    # Estymacja ocen wszystkich kryteriów i podmiana znanej wartości
                    est_evals = front_eval[nearest_idx].copy()
                    est_evals[:, self.objective.obj] = exact_evals

                    # Normalizacja wyników do przedziału [0, 1]
                    F_norm = (est_evals - ideal) / denom

                    # Obliczanie d1 (Zbieżność) - iloczyn skalarny wektora wyniku z wektorem wag
                    d1 = np.dot(F_norm, w)

                    # Obliczanie d2 (Różnorodność) - odległość prostopadła (błąd odchylenia od wektora)
                    projection = d1[:, np.newaxis] * w
                    d2 = np.linalg.norm(F_norm - projection, axis=1)

                    # Ostateczny wynik PBI
                    return d1 + (theta * d2)

                parent_scalar = calc_pbi(temp_pop, temp_pop_eval)
                arg_sort = parent_scalar.argsort()

        # Sort population to determine affinity (lower evaluation = better rank)
        if not use_scalarization:
            arg_sort = temp_pop_eval.argsort()

        indices: List[int] = []
        better: List[np.ndarray] = []
        better_eval: List[float] = []
        evaluation_counter: int = 0

        if self.strategy == 'all_best':
            # 'all_best' strategy: Pool all clones together and select the top N individuals overall
            all_clones: Optional[np.ndarray] = None
            all_clones_eval: Optional[np.ndarray] = None

            for rank, arg in enumerate(arg_sort):
                # Number of clones is inversely proportional to rank
                clone_num = max(int(self.nclone / (rank + 1) + 0.5), 1)
                # Clone and mutate
                norm_rank = rank / max(1, len(arg_sort) - 1)
                clones = np.array([self._mutate(temp_pop[arg], pattern, norm_rank) for _ in range(clone_num)])
                # Filter out clones that did not change
                clones = clones[np.any(clones != temp_pop[arg], axis=1)]

                if clones.shape[0] > 0:
                    clones = self.repair.do(clones)
                    clones_eval = self.objective.evaluate(clones)
                    evaluation_counter += clones.shape[0]

                    if all_clones is None:
                        all_clones = clones
                        all_clones_eval = clones_eval
                    else:
                        all_clones = np.vstack([all_clones, clones])
                        all_clones_eval = np.append(all_clones_eval, clones_eval)

            if all_clones is not None:
                # Combine original population with all generated clones
                all_clones = np.vstack([all_clones, temp_pop])
                all_clones_eval = np.append(all_clones_eval, temp_pop_eval)
                # Select the absolute best individuals to form the new population
                if use_scalarization:
                    all_scalar = calc_pbi(all_clones, all_clones_eval)
                    final_sort = all_scalar.argsort()
                else:
                    final_sort = all_clones_eval.argsort()
                temp_pop[:, :] = all_clones[final_sort[:temp_pop.shape[0]], :]
                temp_pop_eval[:] = all_clones_eval[final_sort[:temp_pop_eval.shape[0]]]

        else:
            # 'base' strategy: Tournament between a single parent and its own clones
            for rank, arg in enumerate(arg_sort):
                clone_num = max(int(self.nclone / (rank + 1) + 0.5), 1)
                norm_rank = rank / max(1, len(arg_sort) - 1)

                clones = np.array([self._mutate(temp_pop[arg], pattern, norm_rank) for _ in range(clone_num)])
                clones = clones[np.any(clones != temp_pop[arg], axis=1)]

                if clones.shape[0] > 0:
                    clones = self.repair.do(clones)
                    clones_eval = self.objective.evaluate(clones)
                    evaluation_counter += clones.shape[0]
                    # Find the best clone among the generated batch
                    if use_scalarization:
                        scalar_c = calc_pbi(clones, clones_eval)
                        argmin = scalar_c.argmin()
                        if scalar_c[argmin] < parent_scalar[arg]:
                            indices.append(arg)
                            better.append(clones[argmin])
                            better_eval.append(clones_eval[argmin])
                    else:
                        argmin = clones_eval.argmin()
                        if clones_eval[argmin] < temp_pop_eval[arg]:
                            indices.append(arg)
                            better.append(clones[argmin])
                            better_eval.append(clones_eval[argmin])

            if len(better) > 0:
                temp_pop[indices] = np.stack(better)
                temp_pop_eval[indices] = np.stack(better_eval)

        # Receptor Editing (Suppression): Replace worst individuals with random new ones
        d = int(pop.shape[0] * self.sup)
        if d > 0:
            if use_scalarization:
                inds = parent_scalar.argsort()[-d:]  # Najgorsi trafiają do kasacji
            else:
                inds = temp_pop_eval.argsort()[-d:]

            pop_sup = np.zeros((inds.shape[0], self.objective.n_var))
            for i in range(inds.shape[0]):

                # Zamiast 'uniform_mutate', używamy inteligentnego krzyżowania osobników z Frontu
                if global_state is not None and len(global_state.get('front', [])) >= 2:
                    front_archive = global_state['front']

                    # Losujemy 2 unikalnych rodziców z globalnego archiwum
                    idx1, idx2 = np.random.choice(len(front_archive), 2, replace=False)
                    p1, p2 = front_archive[idx1], front_archive[idx2]

                    # Krzyżowanie arytmetyczne (współczynnik alpha losowy dla każdego genu)
                    alpha = np.random.rand(self.objective.n_var)
                    hybrid_genes = alpha * p1 + (1.0 - alpha) * p2
                else:
                    # Fallback awaryjny - jeśli front jest jeszcze za mały
                    hybrid_genes = self._create_individual_uniform(self.objective.bounds)

                # Aplikujemy hybrydowe geny TYLKO w dozwolonych miejsach (pattern)
                pop_sup[i] = np.where(pattern, hybrid_genes, temp_pop[inds[i]])

            # Naprawa ewentualnych wyjść poza dopuszczalne bounds (dzięki naprawie z Player)
            pop_sup = self.repair.do(pop_sup)
            pop_eval_sup = self.objective.evaluate(pop_sup)
            evaluation_counter += pop_sup.shape[0]

            temp_pop[inds, :] = pop_sup[:, :]
            temp_pop_eval[inds] = pop_eval_sup[:]

        return temp_pop, temp_pop_eval, evaluation_counter

    def _mutate(self, ind: np.ndarray, pattern: np.ndarray, norm_rank: float) -> np.ndarray:
        """Applies hypermutation operators based on defined probabilities."""
        a, b, sigma = self.mutate_args
        r = np.random.random()

        if r < a:
            ind = self._uniform_mutate(ind, pattern, self.objective.bounds)
        elif r < b:
            dynamic_sigma = sigma * (1.0 + 4.0 * norm_rank)
            ind = self._gaussian_mutate(ind, pattern, self.objective.bounds, dynamic_sigma)
        else:
            ind = self._bound_mutate(ind, pattern, self.objective.bounds)
        return ind

    @staticmethod
    def _uniform_mutate(individual: np.ndarray, pattern: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Highly optimized, vectorized uniform mutation."""
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind

        r = np.random.random(s) < (1.0 / s)
        mutate_mask = np.zeros_like(pattern, dtype=bool)
        mutate_mask[pattern] = r
        # r = np.random.random(pattern.shape) < (1.0 / s)
        # mutate_mask = np.logical_and(pattern, r)

        if not np.any(mutate_mask):
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            mutate_mask[k] = True

        bounds_arr = np.array(bounds)
        a = bounds_arr[mutate_mask, 0]
        b = bounds_arr[mutate_mask, 1]
        ind[mutate_mask] = np.random.uniform(a, b)
        return ind



    @staticmethod
    def _bound_mutate(individual: np.ndarray, pattern: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Highly optimized, vectorized boundary mutation."""
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind

        r = np.random.random(s) < (1.0 / s)
        mutate_mask = np.zeros_like(pattern, dtype=bool)
        mutate_mask[pattern] = r

        # r = np.random.random(pattern.shape) < (1.0 / s)
        # mutate_mask = np.logical_and(pattern, r)

        if not np.any(mutate_mask):
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            mutate_mask[k] = True

        bounds_arr = np.array(bounds)
        a = bounds_arr[mutate_mask, 0]
        b = bounds_arr[mutate_mask, 1]

        num_mutated = np.sum(mutate_mask)
        current_vals = ind[mutate_mask]

        r1 = np.random.random(num_mutated)
        r2 = np.random.uniform(0, 1, num_mutated)

        val_lower = a + (current_vals - a) * r2
        val_upper = current_vals + (b - current_vals) * r2
        ind[mutate_mask] = np.where(r1 < 0.5, val_lower, val_upper)
        return ind

    @staticmethod
    def _gaussian_mutate(individual: np.ndarray, pattern: np.ndarray, bounds: List[Tuple[float, float]], sigma: float) -> np.ndarray:
        """Highly optimized, vectorized Gaussian noise mutation."""
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind

        r = np.random.random(s) < (1.0 / s)
        mutate_mask = np.zeros_like(pattern, dtype=bool)
        mutate_mask[pattern] = r

        # r = np.random.random(pattern.shape) < (1.0 / s)
        # mutate_mask = np.logical_and(pattern, r)

        if not np.any(mutate_mask):
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            mutate_mask[k] = True

        bounds_arr = np.array(bounds)
        a = bounds_arr[mutate_mask, 0]
        b = bounds_arr[mutate_mask, 1]

        noise = sigma * (b - a) * np.random.randn(np.sum(mutate_mask))
        ind[mutate_mask] = np.clip(ind[mutate_mask] + noise, a, b)
        return ind

    def create_population(self) -> np.ndarray:
        """
        Inicjalizacja populacji za pomocą Latin Hypercube Sampling (LHS).
        Rozwiązuje problem pustych 'dziur' i 'skupisk' w przestrzeni decyzyjnej
        poprzez zagwarantowanie równomiernego podziału dziedziny każdego wymiaru.
        """
        # Przygotowanie pustej macierzy [pop_size, n_vars] w przedziale [0, 1]
        pop_size, n_vars = self.npop, self.objective.n_var
        samples = np.empty((pop_size, n_vars))

        # Generowanie jednostajnej hiperkostki
        for i in range(n_vars):
            # 1. Losowa permutacja przypisująca osobnika do "koszyka" (siatki na szachownicy)
            bins = np.random.permutation(pop_size)

            # 2. Losowe mikrostrojenie (szum) ściśle wewnątrz przydzielonego koszyka
            offset = np.random.uniform(0.0, 1.0, size=pop_size)

            # 3. Złożenie wartości w jeden wymiar i normalizacja do przedziału [0.0, 1.0]
            samples[:, i] = (bins + offset) / pop_size

        # Skalowanie wygenerowanej hiperkostki [0, 1] do rzeczywistych ograniczeń problemu
        if hasattr(self.objective, 'bounds') and self.objective.bounds is not None:
            bounds_arr = np.array(self.objective.bounds)
            lower_bounds = bounds_arr[:, 0]
            upper_bounds = bounds_arr[:, 1]

            # Równanie prostej interpolacji: Min + Wartość_Z_Zakresu_0_1 * (Max - Min)
            samples = lower_bounds + samples * (upper_bounds - lower_bounds)

        return samples
