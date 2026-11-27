"""
Klasyfikacja zmiennych decyzyjnych na zmienne kompromisu (pozycja na froncie) i zbieżności
na podstawie archiwum GlobalStorage - bez dodatkowych ewaluacji.

Schemat:
  1. udziały p_k = f~_k / sum_j f~_j (f~ = min-max) - usuwają wspólny mnożnik odległości od frontu,
  2. wynik zmiennej s_j = max_k max(|Spearman(x_j, p_k)|, sqrt(skorygowane eta^2(p_k | x_j))),
  3. próg z permutacji wierszy front_eval (Bonferroni po zmiennych),
  4. krok warunkowy: kandydat zostaje tylko, jeśli niesie informację o pozycji PONAD już wybrane
     zmienne kompromisu (reszta kNN x_j - E[x_j | wybrane]); usuwa zmienne powiązane (UF/LZ)
     i większość fałszywych trafień z pokrewieństwa punktów archiwum,
  5. głosowanie w czasie (EMA) z histerezą,
  6. przydział: zmienne zbieżności rozłącznie między graczy, zmienne kompromisu każdemu graczowi
     z prawdop. p_div, a każda ma co najmniej jednego właściciela.
"""
from typing import Optional, Union

import numpy as np

from agamoo_ray.utils import assigning_gens


# ---------------------------------------------------------------- miary związku
def _ranks(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a, axis=0), axis=0).astype(float)


def _spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|rho| dla wszystkich par kolumn: (d, m)."""
    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean(axis=0)
    ry -= ry.mean(axis=0)
    den = np.sqrt((rx ** 2).sum(axis=0))[:, None] * np.sqrt((ry ** 2).sum(axis=0))[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.abs(np.where(den > 0, rx.T @ ry / np.where(den > 0, den, 1.0), 0.0))


def _eta_adj(x: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """sqrt(skorygowane eta^2) dla par kolumn: (d, m). Kosze kwantylowe po x."""
    n = x.shape[0]
    yc = y - y.mean(axis=0)
    tot = (yc ** 2).sum(axis=0)
    eta2 = np.zeros((x.shape[1], y.shape[1]))
    for j in range(x.shape[1]):
        edges = np.quantile(x[:, j], np.linspace(0, 1, n_bins + 1)[1:-1])
        bins = np.searchsorted(edges, x[:, j])
        cnt = np.bincount(bins, minlength=n_bins).astype(float)
        nz = cnt > 0
        for k in range(y.shape[1]):
            if tot[k] > 1e-12:
                s = np.bincount(bins, weights=yc[:, k], minlength=n_bins)
                eta2[j, k] = (s[nz] ** 2 / cnt[nz]).sum() / tot[k]
    # korekta obciążenia: przy braku związku E[eta^2] ~ (b-1)/(n-1)
    eta2 = 1.0 - (1.0 - np.clip(eta2, 0.0, 1.0)) * (n - 1) / max(n - n_bins, 1)
    return np.sqrt(np.clip(eta2, 0.0, 1.0))


def shares(front_eval: np.ndarray) -> np.ndarray:
    f = np.asarray(front_eval, float)
    lo, hi = f.min(axis=0), f.max(axis=0)
    fn = (f - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    s = fn.sum(axis=1, keepdims=True)
    return fn / np.where(s > 0, s, 1.0)


def tradeoff_scores(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    """s_j dla kolumn x względem udziałów p: (d,)."""
    n_bins = max(2, min(10, x.shape[0] // 5))
    return np.maximum(_eta_adj(x, p, n_bins).max(axis=1), _spearman(x, p).max(axis=1))


def knn_residual(x: np.ndarray, s: np.ndarray, k: int = 7) -> np.ndarray:
    """x - E[x | s] (kNN, leave-one-out): część x niewyjaśniona przez zmienne s."""
    span = np.ptp(s, axis=0)
    sn = (s - s.min(axis=0)) / np.where(span > 0, span, 1.0)
    d2 = (sn ** 2).sum(axis=1)[:, None] + (sn ** 2).sum(axis=1)[None, :] - 2.0 * sn @ sn.T
    np.fill_diagonal(d2, np.inf)
    k = min(k, x.shape[0] - 1)
    nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
    return x - x[nn].mean(axis=1)


# ---------------------------------------------------------------- jedna klasyfikacja
def classify_tradeoff_vars(front: np.ndarray, front_eval: np.ndarray, n_perm: int = 30, alpha: float = 0.05,
                           min_front: int = 20, max_points: int = 300, conditional: bool = True,
                           k_nn: int = 7, max_tradeoff: Union[int, str, None] = 'auto',
                           rng: Optional[np.random.Generator] = None) -> Optional[np.ndarray]:
    """
    True = zmienna kompromisu. None = brak wiarygodnego podziału (front za mały,
    wszystkie albo żadna zmienna nie przechodzi progu).
    max_tradeoff - limit liczby zmiennych kompromisu; 'auto' = 2*(m-1), None = bez limitu.
    Pominięta zmienna kompromisu trafia po prostu do przydziału jak dotąd (błąd neutralny),
    a fałszywe trafienie odcina zmienną zbieżności od graczy (błąd szkodliwy) - stąd limit.
    """
    if front is None or len(front) < min_front:
        return None
    rng = rng if rng is not None else np.random.default_rng(np.random.randint(2 ** 31))
    x, f = np.asarray(front, float), np.asarray(front_eval, float)
    if x.shape[0] > max_points:                               # koszt: O(n^2) w kroku kNN
        idx = rng.choice(x.shape[0], max_points, replace=False)
        x, f = x[idx], f[idx]
    n, d = x.shape
    p = shares(f)
    if max_tradeoff == 'auto':
        max_tradeoff = 2 * (f.shape[1] - 1)

    s = tradeoff_scores(x, p)
    null = np.concatenate([tradeoff_scores(x, p[rng.permutation(n)]) for _ in range(n_perm)])
    thr = np.quantile(null, 1.0 - alpha / d)
    cand = np.flatnonzero(s > thr)

    if conditional and cand.size > 1:
        cand = cand[np.argsort(-s[cand])]
        sel = [int(cand[0])]
        for j in cand[1:]:                                    # w przód: wnosi coś ponad wybrane?
            r = knn_residual(x[:, j], x[:, sel], k_nn)
            if tradeoff_scores(r[:, None], p)[0] > thr:
                sel.append(int(j))
        for j in sorted(sel, key=lambda t: s[t]):             # wstecz: od najsłabszej
            others = [t for t in sel if t != j]
            if others:
                r = knn_residual(x[:, j], x[:, others], k_nn)
                if tradeoff_scores(r[:, None], p)[0] <= thr:
                    sel.remove(j)
        cand = np.array(sel, dtype=int)

    if max_tradeoff is not None and cand.size > max_tradeoff:  # limit: najsilniejsze wg s_j
        cand = cand[np.argsort(-s[cand])][:max_tradeoff]

    lab = np.zeros(d, dtype=bool)
    lab[cand] = True
    return None if (lab.all() or not lab.any()) else lab


# ---------------------------------------------------------------- stan w czasie + przydział
class TradeoffClassifier:
    """
    Trzymany w GlobalStorage. update() wywoływane przy każdym przeliczeniu wzorców (change_iter).

    Histereza na EMA etykiet (waga nowego głosu `ema_w`):
      zmienna wchodzi do grupy kompromisu, gdy ema > enter, wychodzi, gdy ema < exit_.
      Przy 0.3 / 0.5 / 0.3: wejście po 2 trafieniach z rzędu, a zmienna stale trafiana
      wypada dopiero po 3-4 kolejnych pudłach (pojedyncze pudło jej nie usuwa).
    """

    def __init__(self, nvars: int, p_div: float = 0.2, ema_w: float = 0.3, enter: float = 0.5,
                 exit_: float = 0.3, min_votes: int = 3, **classify_kw):
        if not 0.0 <= exit_ < enter <= 1.0:
            raise ValueError("TradeoffClassifier: wymagane 0 <= exit_ < enter <= 1")
        self.nvars, self.p_div, self.ema_w = nvars, p_div, ema_w
        self.enter, self.exit_, self.min_votes = enter, exit_, min_votes
        self.classify_kw = classify_kw
        self.reset()

    def reset(self) -> None:
        """Wywołać przy zmianie środowiska (stare etykiety dotyczą innego problemu)."""
        self.ema = np.zeros(self.nvars)
        self.active = np.zeros(self.nvars, dtype=bool)
        self.votes = 0
        self.last: Optional[np.ndarray] = None

    def update(self, front: np.ndarray, front_eval: np.ndarray) -> np.ndarray:
        if front is None or len(front) < self.classify_kw.get('min_front', 20):
            return self.active                                 # za mało danych: bez głosu
        lab = classify_tradeoff_vars(front, front_eval, **self.classify_kw)
        self.last = lab
        vote = lab.astype(float) if lab is not None else np.zeros(self.nvars)   # brak podziału = głos "0"
        self.ema = (1.0 - self.ema_w) * self.ema + self.ema_w * vote
        self.votes += 1
        self.active = np.where(self.active, self.ema >= self.exit_, self.ema > self.enter)
        return self.active

    @property
    def has_split(self) -> bool:
        return self.votes >= self.min_votes and self.active.any() and not self.active.all()

    def patterns(self, nobjs: int) -> np.ndarray:
        if not self.has_split:
            return assigning_gens(self.nvars, nobjs)
        conv = np.flatnonzero(~self.active)
        div = np.flatnonzero(self.active)
        pat = np.zeros((nobjs, self.nvars), dtype=bool)
        pat[:, conv] = assigning_gens(conv.size, nobjs)        # rozłącznie, każdy gracz >= 1 zmienna
        own = np.random.rand(nobjs, div.size) < self.p_div
        orphan = ~own.any(axis=0)                             # każda zmienna kompromisu ma właściciela
        own[np.random.randint(nobjs, size=int(orphan.sum())), np.flatnonzero(orphan)] = True
        pat[:, div] = own
        return pat
