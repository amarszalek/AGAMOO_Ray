import numpy as np
from agamoo_ray import Objective


def decode_population(population, min_weight=0.02, max_weight=0.15, max_delta=0.5):
    """
    Mapuje macierz genotypów populacji na macierz docelowych wag portfela i wektor progów delta.
    Wykonywane całkowicie wektorowo (z wyjątkiem rzadkich przypadków brzegowych) dla maksymalnej wydajności.

    :param population: Tablica 2D (pop_size x (N+1)), gdzie N to liczba aktywów
    :param min_weight: Próg wejścia (Floor Constraint)
    :param max_weight: Maksymalny udział waloru (Ceiling Constraint)
    :param max_delta: Maksymalna wartość progu rebalansowania
    :return: target_weights (tablica 2D: pop_size x N), deltas (tablica 1D: pop_size)
    """
    population = np.asarray(population)
    weights_raw = population[:, :-1]
    delta_genes = population[:, -1]

    pop_size, num_assets = weights_raw.shape
    w = np.copy(weights_raw)

    # 1. WSTĘPNA NORMALIZACJA (wierszami)
    row_sums = np.sum(w, axis=1, keepdims=True)

    # Zabezpieczenie przed dzieleniem przez 0 (jeśli wszystkie geny w wierszu to 0)
    zero_rows_mask = (row_sums == 0).flatten()
    if np.any(zero_rows_mask):
        w[zero_rows_mask, :] = 1.0 / num_assets
        row_sums[zero_rows_mask] = 1.0

    w = w / row_sums

    # 2. PROGOWANIE NA ZNORMALIZOWANYCH WAGACH
    w = np.where(w < min_weight, 0.0, w)

    # Zabezpieczenie (Fallback): Czy w każdym wierszu zostało wystarczająco spółek?
    min_required_assets = int(np.ceil(1.0 / max_weight))
    active_counts = np.sum(w > 0, axis=1)
    fallback_mask = active_counts < min_required_assets

    # Naprawa przypadków brzegowych (Fallback)
    if np.any(fallback_mask):
        fallback_indices = np.where(fallback_mask)[0]
        # Pętla tylko dla ułamka populacji, który wymaga twardej naprawy
        for i in fallback_indices:
            top_indices = np.argsort(weights_raw[i])[-min_required_assets:]
            w[i, :] = 0.0
            w[i, top_indices] = weights_raw[i, top_indices]
            # Normalizacja wyciągniętych najlepszych genów
            sum_top = np.sum(w[i, top_indices])
            if sum_top > 0:
                w[i, top_indices] /= sum_top
            else:
                w[i, top_indices] = 1.0 / min_required_assets

    # Renormalizacja dla wierszy, które przeszły normalnie (nie były w fallbacku)
    valid_mask = ~fallback_mask
    if np.any(valid_mask):
        valid_sums = np.sum(w[valid_mask], axis=1, keepdims=True)
        w[valid_mask] = w[valid_mask] / np.where(valid_sums == 0, 1.0, valid_sums)

    # 3. ITERACYJNA REDYSTRYBUCJA (Ceiling Constraint - zwektoryzowana wierszami)
    tolerance = 1e-6
    while np.any(w > max_weight + tolerance):
        # Maski dla całej populacji
        capped = w > max_weight
        uncapped = (w <= max_weight) & (w > 0)

        # Obliczamy łączną nadwyżkę dla każdego wiersza (rozmiar: pop_size x 1)
        excess = np.sum(np.where(capped, w - max_weight, 0.0), axis=1, keepdims=True)

        # Wiersze, w których nie ma już gdzie upchać nadwyżki (zabezpieczenie pętli)
        needs_fix = np.any(capped, axis=1)
        if np.any(needs_fix) and not np.any(uncapped[needs_fix]):
            break

        # Ucinamy wagi do wartości granicznej
        w = np.where(capped, max_weight, w)

        # Redystrybucja kapitału proporcjonalnie do pozostałych wag w wierszu
        uncapped_sums = np.sum(np.where(uncapped, w, 0.0), axis=1, keepdims=True)
        safe_sums = np.where(uncapped_sums == 0, 1.0, uncapped_sums)

        # Dodajemy kapitał tylko do miejsc nieobciętych
        w += excess * (np.where(uncapped, w, 0.0) / safe_sums)

    # 4. Wyłuskanie progu delta dla każdego wiersza
    deltas = delta_genes * max_delta

    return w, deltas

#decode_kwargs={'min_weight': 0.02, 'max_weight': 0.15, 'max_delta': 0.5}
class Profit(Objective):
    def __init__(self, num, historical_returns, obj=1, args=None, verbose=False, **decode_kwargs):
        """
        :param historical_returns: Macierz 2D (dni x aktywa) dziennych stóp zwrotu
        """
        obj = obj - 1
        n_var = historical_returns.shape[1]+1
        n_obj = 3
        bounds = list(zip([0.0]*n_var, [1.0]*n_var))
        self.historical_returns = np.asarray(historical_returns)
        self.decode_kwargs = decode_kwargs
        super(Profit, self).__init__(num, n_var, n_obj, bounds, obj, args, verbose)


    def evaluate(self, x):
        target_weights, deltas = decode_population(x, **self.decode_kwargs)
        pop_size = target_weights.shape[0]

        portfolio_values = np.ones(pop_size)
        current_weights = np.copy(target_weights)

        # Pętla po czasie (tego w backteście path-dependent wektoryzować się nie da,
        # ale wewnątrz pętli liczymy wszystko dla wszystkich osobników naraz)
        for t in range(len(self.historical_returns)):
            daily_ret = self.historical_returns[t]  # Wektor 1D: (N,)

            # Zwrot każdego portfela w populacji (Suma iloczynów wierszami)
            # Kształt: (pop_size,)
            portfolio_returns = np.sum(current_weights * daily_ret, axis=1)
            portfolio_values *= (1 + portfolio_returns)

            # Dryf wag we wszystkich portfelach
            # Używamy [:, None] aby podzielić kolumny macierzy 2D przez wektor 1D
            current_weights = current_weights * (1 + daily_ret) / (1 + portfolio_returns)[:, None]

            # Sprawdzenie warunku rebalansowania dla całej populacji
            drifts = np.sum(np.abs(current_weights - target_weights), axis=1)
            rebalance_mask = drifts > deltas

            # Reset wag tylko dla tych portfeli, które przekroczyły swój próg delta
            current_weights[rebalance_mask] = target_weights[rebalance_mask]

        # Zwracamy tablicę ujemnych zysków (do minimalizacji)
        return -portfolio_values


class Cost(Objective):
    def __init__(self, num, historical_returns, min_fee=5.0, c_prop=0.0039, portfolio_capital=10000.0, obj=1,
                 args=None, verbose=False, **decode_kwargs):
        obj = obj - 1
        n_var = historical_returns.shape[1]+1
        n_obj = 3
        bounds = list(zip([0.0] * n_var, [1.0] * n_var))
        self.historical_returns = np.asarray(historical_returns)
        self.c_prop = c_prop
        self.decode_kwargs = decode_kwargs
        self.c_min_pct = min_fee / portfolio_capital
        super(Cost, self).__init__(num, n_var, n_obj, bounds, obj, args, verbose)

    def evaluate(self, population):
        target_weights, deltas = decode_population(population, **self.decode_kwargs)
        pop_size, num_assets = target_weights.shape

        total_costs = np.zeros(pop_size)
        current_weights = np.copy(target_weights)

        # 1. Koszt startowy portfela (Inicjalizacja)
        # Obliczamy koszt dla każdej zakupionej pozycji z osobna
        initial_trades = target_weights > 0

        # Funkcja MAX(c_min, trade * c_prop) wyliczona wektorowo dla każdej spółki
        initial_trade_costs = np.where(
            initial_trades,
            np.maximum(self.c_min_pct, target_weights * self.c_prop),
            0.0
        )
        # Sumujemy koszty wszystkich spółek dla każdego portfela (wierszami)
        total_costs += np.sum(initial_trade_costs, axis=1)

        for t in range(len(self.historical_returns)):
            daily_ret = self.historical_returns[t]
            portfolio_returns = np.sum(current_weights * daily_ret, axis=1)

            # Dryf wag w wyniku zmian cen
            current_weights = current_weights * (1 + daily_ret) / (1 + portfolio_returns)[:, None]

            # Decyzja o rebalansowaniu dla całej populacji
            drifts = np.sum(np.abs(current_weights - target_weights), axis=1)
            rebalance_mask = drifts > deltas

            if np.any(rebalance_mask):
                # Obliczamy wielkość zlecenia dla każdej spółki osobno
                # Rozmiar: (liczba_portfeli_rebalansujących, num_assets)
                weight_changes = np.abs(current_weights[rebalance_mask] - target_weights[rebalance_mask])

                # Handlujemy tylko tam, gdzie waga faktycznie się zmieniła
                # (Zabezpieczenie przed błędami zmiennoprzecinkowymi)
                trade_mask = weight_changes > 1e-6

                # Wektorowe wyliczenie kosztów z funkcją MAX na poszczególnych aktywach
                trade_costs = np.where(
                    trade_mask,
                    np.maximum(self.c_min_pct, weight_changes * self.c_prop),
                    0.0
                )

                # Dodajemy zsumowane koszty poszczególnych zleceń do całkowitego kosztu portfela
                total_costs[rebalance_mask] += np.sum(trade_costs, axis=1)

                # Przywrócenie wag
                current_weights[rebalance_mask] = target_weights[rebalance_mask]

        return total_costs


class CVaR(Objective):
    def __init__(self, num, historical_returns, num_paths=5000, horizon=252, alpha=0.05, obj=1,
                 args=None, verbose=False, **decode_kwargs):
        obj = obj - 1
        n_var = historical_returns.shape[1] + 1
        n_obj = 3
        bounds = list(zip([0.0] * n_var, [1.0] * n_var))
        self.historical_returns = np.asarray(historical_returns)
        self.num_paths = num_paths
        self.horizon = horizon
        self.alpha = alpha
        self.mean_returns = np.mean(self.historical_returns, axis=0)
        self.cov_matrix = np.cov(self.historical_returns, rowvar=False)
        self.num_assets = len(self.mean_returns)
        self.decode_kwargs = decode_kwargs

        try:
            self.L = np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            # Jeśli macierz nie jest w pełni dodatnio określona, dodajemy mikroskopijny
            # szum na przekątną (jitter), co rozwiązuje problem numeryczny bez wpływu na wyniki.
            jitter = 1e-8 * np.eye(self.num_assets)
            self.L = np.linalg.cholesky(self.cov_matrix + jitter)
        super(CVaR, self).__init__(num, n_var, n_obj, bounds, obj, args, verbose)

    def evaluate(self, population):
        target_weights, _ = decode_population(population, **self.decode_kwargs)
        pop_size = target_weights.shape[0]

        # Macierz przechowująca ostateczną wartość (dla całej populacji x liczba ścieżek)
        final_portfolio_values = np.ones((pop_size, self.num_paths))

        # Idziemy po ścieżkach Monte Carlo
        for i in range(self.num_paths):
            # 1. Generujemy całą ścieżkę dla rynku naraz (Horyzont x Aktywa)
            Z = np.random.normal(0, 1, (self.horizon, self.num_assets))
            correlated_shocks = Z @ self.L.T
            # Zwroty dzienne rynkowe na tej ścieżce
            daily_returns = self.mean_returns + correlated_shocks

            # 2. BŁYSKAWICZNA SYMULACJA DLA CAŁEJ POPULACJI
            # daily_returns to (horizon, num_assets)
            # target_weights.T to (num_assets, pop_size)
            # Wynik port_rets to macierz (horizon, pop_size) ze zwrotami wszystkich
            # portfeli w każdym dniu tej jednej symulowanej ścieżki!
            port_rets = daily_returns @ target_weights.T

            # 3. Skumulowany zwrot ścieżki (iloczyn w kolumnach po czasie - axis=0)
            # Wynik: Wektor 1D o rozmiarze (pop_size,)
            path_values = np.prod(1 + port_rets, axis=0)

            # Zapisujemy wygenerowane wartości w i-tej kolumnie
            final_portfolio_values[:, i] = path_values

        # Obliczenie CVaR dla każdego portfela naraz
        # Sortujemy wartości portfeli po osi ścieżek (axis=1) od najgorszej do najlepszej
        sorted_values = np.sort(final_portfolio_values, axis=1)

        cutoff_index = int(self.num_paths * self.alpha)

        # Średnia z najgorszych (np. 5%) ścieżek dla każdego osobnika w populacji
        tail_scenarios = sorted_values[:, :cutoff_index]
        cvars = 1.0 - np.mean(tail_scenarios, axis=1)

        # Fallback - gigantyczna kara dla portfeli, które zbugowały się do 0 aktywów
        empty_mask = np.sum(target_weights, axis=1) == 0
        if np.any(empty_mask):
            cvars[empty_mask] = 999999.0

        return cvars