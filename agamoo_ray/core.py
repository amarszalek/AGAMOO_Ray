import time
import ray
import numpy as np
import logging
import asyncio
import traceback
from tqdm.auto import tqdm
from agamoo_ray.utils import get_not_dominated, front_suppression, assigning_gens

logger = logging.getLogger(__name__)


class AGAMOO:
    def __init__(self, max_eval, change_iter, next_iter, max_front, max_front_tol=0,
                 init_pop='separate', front_f=None, verbose=False):

        self.max_eval = max_eval
        self.change_iter = change_iter
        self.next_iter = next_iter
        self.max_front = max_front
        self.max_front_tol = max_front_tol
        self.init_pop = init_pop
        self.front_f = front_f
        self.verbose = verbose

        self.players = []
        self.evaluator = None
        self.storage = None
        self.results = None

        self.nobjs = 0
        self.nvars = 0

    def init_players(self, players, evaluator, repair=None):
        """
        Przyjmuje listę aktorów (Players).
        UWAGA: W tej wersji 'players' to lista uchwytów do aktorów (ActorHandle),
        a nie instancji klas.
        """
        self.players = players
        self.evaluator = evaluator
        self.repair = repair
        # Rejestrujemy graczy w storage
        ray.get(self.storage.set_players.remote(players))
        ray.get(self.storage.set_evaluator.remote(evaluator))


    def create_storage(self, nvars, nobjs, num_cpus=1):
        """
        Tworzy instancję GlobalStorage. Należy to wywołać przed utworzeniem graczy,
        aby przekazać im uchwyt.
        """
        self.nvars = nvars
        self.nobjs = nobjs

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self.storage = GlobalStorage.options(num_cpus=num_cpus).remote(
            nvars, nobjs, self.max_eval, self.change_iter, self.next_iter,
            self.max_front, self.max_front_tol, self.front_f
        )
        return self.storage

    def start_optimize(self, tqdm_disable=False):
        if not self.players:
            raise ValueError("Players list is empty. Initialize players first.")
        if not self.storage:
            raise ValueError("Storage not created. Call create_storage() first.")

        # Reset stanu (na wypadek ponownego uruchomienia)
        ray.get(self.storage.reset.remote())

        if self.verbose:
            logger.info("Starting optimization with Ray...")

        # Uruchomienie graczy
        for p in self.players:
            p.set_repair.remote(self.repair)
            p.start.remote()

        if self.verbose:
            logger.info("Running players ...")

        # Główna pętla monitorująca (Driver Loop)
        with tqdm(total=self.max_eval, unit='eval', disable=tqdm_disable) as pbar:
            while True:
                # Pobierz status z aktora Storage
                status = ray.get(self.storage.get_status.remote())

                #if self.verbose:
                #    print(f"Get status: {status}")

                # Aktualizacja paska postępu
                current_evals = status['evaluations']
                pbar.n = min(current_evals, self.max_eval)
                pbar.refresh()

                # Sprawdzenie warunku stopu
                if status['stop_flag']:
                    break

                time.sleep(0.5)  # Odświeżanie co 0.5s

        if self.verbose:
            logger.info("Optimization finished.")

        # Zabijamy aktorów graczy, aby zakończyć ich pętle while True
        for p in self.players:
            ray.kill(p)

    def get_results(self, key=None):
        if not self.storage:
            return None

        if key:
            # Pobranie konkretnego klucza (wymaga rozszerzenia Storage o generyczny getter
            # lub pobrania całego słownika)
            snapshot = ray.get(self.storage.get_snapshot.remote())
            return snapshot.get(key)

        return ray.get(self.storage.get_results.remote())


@ray.remote
class GlobalStorage:
    """
    Aktor Ray odpowiedzialny za przechowywanie globalnego stanu optymalizacji,
    zarządzanie frontem Pareto oraz synchronizację iteracji.
    """

    def __init__(self, nvars, nobjs, max_eval, change_iter, next_iter, max_front,
                 max_front_tol=0.0, front_f=None, verbose=False):
        self.nvars = nvars
        self.nobjs = nobjs
        self.max_eval = max_eval
        self.change_iter = change_iter
        self.next_iter = next_iter
        self.max_front = max_front
        self.max_front_tol = max_front_tol
        self.front_f = front_f
        self.players_handles = []  # Uchwyty do aktorów graczy
        self.evaluator_handle = None
        self.verbose = verbose

        # Stan wewnętrzny
        self.reset()

    def set_players(self, players):
        """Rejestruje uchwyty do graczy, aby móc zlecać im obliczenia."""
        self.players_handles = players

    def set_evaluator(self, evaluator):
        """Rejestruje uchwyty do procesów obliczeniowych (Ewaluatorów)."""
        self.evaluator_handle = evaluator

    def reset(self):
        """Resetuje stan do wartości początkowych przed nową optymalizacją."""
        self.front = np.empty((0, self.nvars))
        self.front_eval = np.empty((0, self.nobjs))
        self.best = [None] * self.nobjs

        self.iter_counters = np.zeros(self.nobjs)
        self.evaluations_count = np.zeros(self.nobjs)
        self.evaluations_time = np.zeros(self.nobjs)
        self.repair_time = np.zeros(self.nobjs)

        self.stop_flag = False
        self.min_iter_pop = 0

        # Inicjalizacja wzorców (gens)
        self.patterns = assigning_gens(self.nvars, self.nobjs)

        # Inne metryki
        self.total_evaluations = 0

    def get_snapshot(self):
        """
        Zwraca "migawkę" stanu dla graczy (Players).
        Dzięki Ray Plasma Store, duże tablice (front) są przesyłane przez zero-copy.
        """
        return {
            'front': self.front,
            'front_eval': self.front_eval,
            'best': self.best,
            'iter_counters': self.iter_counters,
            'patterns': self.patterns,
            'next_iter': self.next_iter,
            'stop_flag': self.stop_flag
        }

    def get_status(self):
        """Zwraca status dla paska postępu (Driver)."""
        return {
            'iterations': self.iter_counters,
            'evaluations': self.total_evaluations,
            'stop_flag': self.stop_flag,
            'front_size': len(self.front)
        }

    def get_results(self):
        """Zwraca końcowe wyniki."""
        # Ostatnie filtrowanie przed zwróceniem wyników
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

    async def update(self, data):
        """
        Główna metoda aktualizująca stan na podstawie danych od Gracza.
        """
        try:
            nobj = data['nobj']
            iteration = data.get('iteration', 0)
            if self.verbose:
                logger.info(f"GlobalStorage received update form obj {nobj}")

            # Aktualizacja liczników iteracji
            self.iter_counters[nobj] = iteration

            # Logika zmiany wzorców (gens) co change_iter
            min_iter = np.min(self.iter_counters)
            if min_iter - self.min_iter_pop >= self.change_iter:
                self.patterns = assigning_gens(self.nvars, self.nobjs)
                self.min_iter_pop = min_iter

            # Jeśli to tylko heartbeat (iter_flag=True), kończymy
            if data.get('iter_flag', False):
                return

            # 2. Pobranie danych populacji
            pop = data['population']
            pop_eval_partial = data['population_eval']
            neval = data['evaluation_counter']

            # Aktualizacja statystyk czasu i liczby ewaluacji
            if neval > 0:
                self.evaluations_count[nobj] = neval
            self.total_evaluations = np.min(self.evaluations_count)

            # 3. Aktualizacja Best Solution dla danej funkcji celu
            if len(pop_eval_partial) > 0:
                best_idx = np.argmin(pop_eval_partial)
                self.best[nobj] = pop[best_idx].copy()

            pop_eval = np.zeros((pop.shape[0], self.nobjs))
            pop_eval[:, nobj] = pop_eval_partial

            futures = []
            target_objs = []
            for i in range(self.nobjs):
                if i != nobj:
                    # Zlecamy obliczenie evaluatorowi
                    futures.append(self.evaluator_handle.evaluate.remote(pop, i))
                    target_objs.append(i)

            # Czekamy na wyniki (non-blocking await w Ray)
            if futures:
                #results = await ray.gather(futures)  # Dostępne w nowszym Ray lub używamy pętli await
                results = await asyncio.gather(*futures)
                # Alternatywnie: results = await asyncio.gather(*futures) jeśli futures są awaitable

                for idx, res in enumerate(results):
                    obj_idx = target_objs[idx]
                    pop_eval[:, obj_idx] = res
                    self.evaluations_count[obj_idx] += pop.shape[0]
            self.total_evaluations = np.min(self.evaluations_count)

            # Dołączamy nową populację do istniejącego frontu
            if len(self.front) == 0:
                self.front = pop
                self.front_eval = pop_eval
            else:
                self.front = np.vstack([self.front, pop])
                self.front_eval = np.vstack([self.front_eval, pop_eval])

            # Filtrowanie niezdominowanych
            if len(self.front) > 1:
                mask = get_not_dominated(self.front_eval)
                self.front = self.front[mask]
                self.front_eval = self.front_eval[mask]

            # Dodatkowe filtrowanie użytkownika (front_f)
            if self.front_f is not None and len(self.front) > 0:
                mask = self.front_f(self.front_eval)
                self.front = self.front[mask]
                self.front_eval = self.front_eval[mask]

            # Suppression (ograniczanie rozmiaru frontu)
            limit = self.max_front
            if self.max_front_tol > 0:
                limit = int(self.max_front * (1.0 + self.max_front_tol))

            if len(self.front) > limit:
                mask = front_suppression(self.front_eval, self.max_front)
                self.front = self.front[mask]
                self.front_eval = self.front_eval[mask]

            # Sprawdzenie warunku stopu
            if self.max_eval > 0 and self.total_evaluations >= self.max_eval:
                self.stop_flag = True

        except Exception as e:
            print(f"GlobalStorage update error: {e}")
            traceback.print_exc()


