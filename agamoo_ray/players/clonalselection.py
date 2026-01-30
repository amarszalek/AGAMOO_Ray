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
                clones = np.array([self._mutate(temp_pop[arg], pattern) for _ in range(clone_num)])
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

    def _mutate(self, ind, pattern):
        a, b, sigma = self.mutate_args
        r = np.random.random()
        if r < a:
            ind = self._uniform_mutate(ind, pattern, self.objective.bounds)
        elif r < b:
            ind = self._gaussian_mutate(ind, pattern, self.objective.bounds, sigma)
        else:
            ind = self._bound_mutate(ind, pattern, self.objective.bounds)
        return ind

    @staticmethod
    def _uniform_mutate(individual, pattern, bounds):
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