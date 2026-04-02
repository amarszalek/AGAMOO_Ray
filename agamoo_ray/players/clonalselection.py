import numpy as np
import ray
import time
from copy import deepcopy
from agamoo_ray.player import Player


@ray.remote
class ClonalSelection(Player):
    def __init__(self, num, npop, player_param, objective, storage_actor,
                 gens='pattern', exchange='front_random', verbose=False, init_pop=None):

        # Pobieranie parametrów specyficznych dla ClonalSelection
        self.nclone = player_param.get('nclone', 15)
        self.mutate_args = tuple(player_param.get('mutate_args', [0.45, 0.9, 0.01]))
        self.sup = player_param.get('sup', 0.0)
        self.strategy = player_param.get('strategy', 'base')

        # Inicjalizacja klasy bazowej Player (wersja Ray)
        super().__init__(num, npop, objective, storage_actor,
                         gens, exchange, verbose, init_pop)

    def step(self, pop, pop_eval, pattern):
        temp_pop = deepcopy(pop)
        temp_pop_eval = deepcopy(pop_eval)
        arg_sort = temp_pop_eval.argsort()
        indices = []
        better = []
        better_eval = []
        evaluation_counter = 0

        if self.strategy == 'all_best':
            all_clones = None
            all_clones_eval = None
            for rank, arg in enumerate(arg_sort):
                clone_num = max(int(self.nclone / (rank + 1) + 0.5), 1)
                clones = np.array([self._mutate(temp_pop[arg], pattern) for _ in range(clone_num)])

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
                all_clones = np.vstack([all_clones, temp_pop])
                all_clones_eval = np.append(all_clones_eval, temp_pop_eval)
                arg_sort = all_clones_eval.argsort()

                temp_pop[:, :] = all_clones[arg_sort[:temp_pop.shape[0]], :]
                temp_pop_eval[:] = all_clones_eval[arg_sort[:temp_pop_eval.shape[0]]]

        else:  # base strategy
            for rank, arg in enumerate(arg_sort):
                clone_num = max(int(self.nclone / (rank + 1) + 0.5), 1)
                norm_rank = rank / max(1, len(arg_sort) - 1)
                clones = np.array([self._mutate(temp_pop[arg], pattern, norm_rank) for _ in range(clone_num)])
                clones = clones[np.any(clones != temp_pop[arg], axis=1)]

                if clones.shape[0] > 0:
                    clones = self.repair.do(clones)
                    clones_eval = self.objective.evaluate(clones)

                    evaluation_counter += clones.shape[0]
                    argmin = clones_eval.argmin()

                    if clones_eval[argmin] < temp_pop_eval[arg]:
                        indices.append(arg)
                        better.append(clones[argmin])
                        better_eval.append(clones_eval[argmin])

            if len(better) > 0:
                better = np.stack(better)
                better_eval = np.stack(better_eval)
                temp_pop[indices] = better
                temp_pop_eval[indices] = better_eval

        # Obsługa parametru 'sup' (suppression/diversity injection)
        d = int(pop.shape[0] * self.sup)
        if d > 0:
            inds = temp_pop_eval.argsort()[-d:]
            pop_sup = np.zeros((inds.shape[0], self.objective.n_var))
            for i in range(inds.shape[0]):
                pop_sup[i] = pop_sup[i] + np.where(pattern,
                                                   self._create_individual_uniform(self.objective.bounds),
                                                   temp_pop[inds[i]])

            pop_sup = self.repair.do(pop_sup)
            pop_eval_sup = self.objective.evaluate(pop_sup)

            evaluation_counter += pop_sup.shape[0]
            temp_pop[inds, :] = pop_sup[:, :]
            temp_pop_eval[inds] = pop_eval_sup[:]

        return temp_pop, temp_pop_eval, evaluation_counter

    def _mutate(self, ind, pattern, norm_rank):
        a, b, sigma = self.mutate_args
        a = a * norm_rank
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
    def _uniform_mutate(individual, pattern, bounds):
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind

        # 1. Tworzenie maski mutacji (szansa 1/s na gen z patternu)
        r = np.random.random(pattern.shape) < (1.0 / s)
        mutate_mask = np.logical_and(pattern, r)

        # 2. Fallback: Jeśli nic się nie wylosowało, mutujemy jeden losowy gen
        if not np.any(mutate_mask):
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            mutate_mask[k] = True

        # 3. Wektoryzacja: Pobieramy granice tylko dla mutowanych genów
        bounds_arr = np.array(bounds)
        a = bounds_arr[mutate_mask, 0]
        b = bounds_arr[mutate_mask, 1]

        # 4. Aplikacja mutacji - np.random.uniform obsługuje tablice dla 'low' i 'high'!
        ind[mutate_mask] = np.random.uniform(a, b)

        return ind


    @staticmethod
    def _uniform_mutate_old(individual, pattern, bounds):
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind
        r = np.random.random(pattern.shape) < 1 / s
        r = np.logical_and(pattern, r)
        indx = np.where(r)[0]
        if len(indx) > 0:
            for k in indx:
                a = bounds[k][0]
                b = bounds[k][1]
                ind[k] = np.random.uniform(a, b)
        else:
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            a = bounds[k][0]
            b = bounds[k][1]
            ind[k] = np.random.uniform(a, b)
        return ind

    @staticmethod
    def _bound_mutate(individual, pattern, bounds):
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind

        # 1. Tworzenie maski mutacji
        r = np.random.random(pattern.shape) < (1.0 / s)
        mutate_mask = np.logical_and(pattern, r)

        # 2. Fallback
        if not np.any(mutate_mask):
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            mutate_mask[k] = True

        # 3. Pobieranie danych do obliczeń wektorowych
        bounds_arr = np.array(bounds)
        a = bounds_arr[mutate_mask, 0]
        b = bounds_arr[mutate_mask, 1]

        num_mutated = np.sum(mutate_mask)
        current_vals = ind[mutate_mask]

        # 4. Wektorowe losowanie parametrów r1 i r2 dla wszystkich mutowanych genów naraz
        r1 = np.random.random(num_mutated)
        r2 = np.random.uniform(0, 1, num_mutated)

        # 5. Obliczamy oba warianty równolegle (lewy i prawy skok)
        val_lower = a + (current_vals - a) * r2
        val_upper = current_vals + (b - current_vals) * r2

        # 6. Wybieramy odpowiedni wariant na podstawie r1 < 0.5 (Wektoryzowany odpowiednik if/else)
        ind[mutate_mask] = np.where(r1 < 0.5, val_lower, val_upper)

        return ind


    @staticmethod
    def _bound_mutate_old(individual, pattern, bounds):
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind
        r = np.random.random(pattern.shape) < 1 / s
        r = np.logical_and(pattern, r)
        indx = np.where(r)[0]
        if len(indx) > 0:
            for k in indx:
                a = bounds[k][0]
                b = bounds[k][1]
                r1 = np.random.random()
                r2 = np.random.uniform(0, 1)
                if r1 < 0.5:
                    ind[k] = a + (ind[k] - a) * r2
                else:
                    ind[k] = (b - ind[k]) * r2 + ind[k]
        else:
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            a = bounds[k][0]
            b = bounds[k][1]
            r1 = np.random.random()
            r2 = np.random.uniform(0, 1)
            if r1 < 0.5:
                ind[k] = a + (ind[k] - a) * r2
            else:
                ind[k] = (b - ind[k]) * r2 + ind[k]
        return ind

    @staticmethod
    def _gaussian_mutate(individual, pattern, bounds, sigma):
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind

        # Prawdopodobieństwo mutacji poszczególnych genów (1/s)
        r = np.random.random(pattern.shape) < (1.0 / s)
        mutate_mask = np.logical_and(pattern, r)

        # Fallback: Jeśli nic się nie wylosowało, mutujemy jeden losowy gen z patternu
        if not np.any(mutate_mask):
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            mutate_mask[k] = True

        # Zamiast pętli 'for k in indx:', robimy wszystko w jednej operacji wektorowej!
        # Konwersja bounds do array (najlepiej zrobić to raz w __init__, ale tu dla spójności)
        bounds_arr = np.array(bounds)

        # Pobieramy tylko te granice, które będą mutowane
        a = bounds_arr[mutate_mask, 0]
        b = bounds_arr[mutate_mask, 1]

        # Wektorowe losowanie szumu Gaussa dla wszystkich wybranych genów naraz
        noise = sigma * (b - a) * np.random.randn(np.sum(mutate_mask))

        # Aplikujemy szum i docinamy do granic (np.clip robi to błyskawicznie i wektorowo)
        ind[mutate_mask] = np.clip(ind[mutate_mask] + noise, a, b)

        return ind
    @staticmethod
    def _gaussian_mutate_old(individual, pattern, bounds, sigma):
        ind = individual.copy()
        s = np.sum(pattern)
        if s == 0:
            return ind
        r = np.random.random(pattern.shape) < 1 / s
        r = np.logical_and(pattern, r)
        indx = np.where(r)[0]
        if len(indx) > 0:
            for k in indx:
                a = bounds[k][0]
                b = bounds[k][1]
                ran = sigma * (b - a) * np.random.randn() + ind[k]
                if a <= ran <= b:
                    ind[k] = ran
                elif ran < a:
                    ind[k] = a
                else:
                    ind[k] = b
        else:
            indx = np.where(pattern)[0]
            k = np.random.choice(indx)
            a = bounds[k][0]
            b = bounds[k][1]
            ran = sigma * (b - a) * np.random.randn() + ind[k]
            if a <= ran <= b:
                ind[k] = ran
            elif ran < a:
                ind[k] = a
            else:
                ind[k] = b
        return ind