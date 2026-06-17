import numpy as np
import logging
from typing import Tuple
import shap
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

logger = logging.getLogger(__name__)

# Attempt to load the optimized C-extension for heavy Pareto operations
CEXT = False
try:
    import agamoo_ray.cutils as cutils
    CEXT = True
except ImportError as e:
    logger.warning("C extension 'cutils' not available. Falling back to NumPy implementations.")
    logger.debug(f"Import error details: {e}")
    CEXT = False


def pairwise_dominance(x: np.ndarray) -> np.ndarray:
    """
    Computes the non-dominated mask using vectorized pairwise comparisons.

    Args:
        x (np.ndarray): Array of evaluated objective values.

    Returns:
        np.ndarray: A boolean mask where True indicates a non-dominated solution.
    """
    worse_or_equal = np.all(x[:, np.newaxis] >= x, axis=2)
    strictly_worse = np.any(x[:, np.newaxis] > x, axis=2)

    is_dominated_matrix = worse_or_equal & strictly_worse

    is_dominated = np.any(is_dominated_matrix, axis=1)

    return ~is_dominated


def get_not_dominated(populations_eval: np.ndarray) -> np.ndarray:
    """
    Filters a population to extract the non-dominated Pareto front.
    Utilizes a C-extension if available for performance, otherwise falls back to NumPy.

    Args:
        populations_eval (np.ndarray): Array of evaluated objective values.

    Returns:
        np.ndarray: A boolean mask representing non-dominated solutions.
    """
    #if CEXT:
    #    mask = np.zeros(populations_eval.shape[0], dtype=np.int32)
    #    cutils.cget_not_dominated(populations_eval, mask)
    #    mask = mask.astype(bool)
    #else:
    mask = pairwise_dominance(populations_eval)
    return mask


def pairwise_distance(x: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean pairwise distance matrix for crowding distance estimation.
    """
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def front_suppression(front: np.ndarray, front_eval: np.ndarray, front_max: int, mode: str = 'objectives') -> np.ndarray:
    """
    Reduces the size of the Pareto front to a specified maximum while maintaining
    diversity using a 'crowding' distance heuristic.

    Args:
        front (np.ndarray): The current Pareto front (solutions).
        front_eval (np.ndarray): The current Pareto front evaluations.
        front_max (int): The maximum allowed size of the front.
        mode (str): Type of suppression ('objectives', 'variables', 'dual_fast' or 'dual_omni').

    Returns:
        np.ndarray: Boolean mask of individuals kept in the suppressed front.
    """

    if mode == 'objectives':
        return _suppression(front_eval, front_max)
    elif mode == 'variables':
        return _suppression(front, front_max)
    elif mode == 'dual_fast':
        if np.random.random()< 0.5:
            mask = _suppression(front_eval, front_max)
        else:
            mask = _suppression(front, front_max)
        return mask
    elif mode == 'dual_omni':
        return _dual_omni_suppression(front, front_eval, front_max)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _suppression(front_eval: np.ndarray, front_max: int) -> np.ndarray:
    """
    Reduces the size of the Pareto front to a specified maximum while maintaining
    diversity using a 'crowding' distance heuristic.

    Args:
        front_eval (np.ndarray): The current Pareto front evaluations.
        front_max (int): The maximum allowed size of the front.

    Returns:
        np.ndarray: Boolean mask of individuals kept in the suppressed front.
    """

    if CEXT:
        mask = np.zeros(front_eval.shape[0], dtype=np.int32)
        cutils.cfront_suppression(front_eval, front_max, mask)
        return mask.astype(bool)

    # NumPy Fallback implementation
    n = front_eval.shape[0] - front_max
    if n <= 0:
        return np.ones(front_eval.shape[0], dtype=bool)

    # Protect ideal boundary points
    ideal = np.argmin(front_eval, axis=0)

    # Normalize front for fair distance calculation
    front_eval_norm = front_eval + np.abs(np.min(front_eval, axis=0)) + 1.0
    front_eval_norm = front_eval_norm / np.max(front_eval_norm, axis=0)

    z = pairwise_distance(front_eval_norm)
    mask = np.ones(front_eval.shape[0], dtype=bool)

    # Ignore upper triangle and diagonal by setting them to a high value
    t = np.tril(z) + np.triu(np.ones_like(z) * 1000000)
    arg = np.argsort(t, axis=None)
    indx_i, indx_j = np.unravel_index(arg, t.shape)

    # Iteratively remove the most crowded individuals
    while n > 0:
        ii = indx_i[0]
        mask[ii] = False

        # Remove dependencies of the deleted point from the distance matrix indices
        tmp = indx_i[indx_i != ii]
        indx_j = indx_j[indx_i != ii]
        indx_i = tmp.copy()

        tmp = indx_j[indx_j != ii]
        indx_i = indx_i[indx_j != ii]
        indx_j = tmp.copy()

        n = n - 1

    # Re-enable the boundary (ideal) points explicitly
    for i in ideal:
        mask[i] = True

    return mask


def _dual_omni_suppression(front: np.ndarray, front_eval: np.ndarray, front_max: int) -> np.ndarray:
    """
    Reduces the size of the Pareto front to a specified maximum while maintaining
    diversity in BOTH objective (F) and decision variable (X) spaces.
    Uses optimized pairwise distance matrices calculated only once.

    Args:
        front (np.ndarray): The current Pareto front variables (X).
        front_eval (np.ndarray): The current Pareto front evaluations (F).
        front_max (int): The maximum allowed size of the front.

    Returns:
        np.ndarray: Boolean mask of individuals kept in the suppressed front.
    """
    n = front_eval.shape[0] - front_max
    if n <= 0:
        return np.ones(front_eval.shape[0], dtype=bool)

    # 1. Zabezpieczenie ekstremów (punktów idealnych) w obu przestrzeniach!
    ideal_f = np.argmin(front_eval, axis=0)
    ideal_x = np.argmin(front, axis=0)
    ideal = np.unique(np.concatenate((ideal_f, ideal_x)))

    # 2. Normalizacja przestrzeni celów (F)
    front_eval_norm = front_eval + np.abs(np.min(front_eval, axis=0)) + 1.0
    front_eval_norm = front_eval_norm / np.max(front_eval_norm, axis=0)

    # 3. Normalizacja przestrzeni zmiennych (X)
    front_norm = front + np.abs(np.min(front, axis=0)) + 1.0
    front_norm = front_norm / np.max(front_norm, axis=0)

    # 4. Obliczenie macierzy odległości dla par (tylko RAZ)
    z_f = pairwise_distance(front_eval_norm)
    z_x = pairwise_distance(front_norm)

    # 5. FUZJA DUALNA (Serce Omni-Optimizera)
    # Odległość dualna to MAKSIMUM z odległości F i X.
    # Jeśli punkty są blisko w F, ale daleko w X, Z_dual będzie duże -> przetrwają.
    z_dual = np.maximum(z_f, z_x)

    mask = np.ones(front_eval.shape[0], dtype=bool)

    # Ignorujemy górny trójkąt i przekątną macierzy odległości
    t = np.tril(z_dual) + np.triu(np.ones_like(z_dual) * 1000000)
    arg = np.argsort(t, axis=None)
    indx_i, indx_j = np.unravel_index(arg, t.shape)

    # 6. Błyskawiczne, iteracyjne usuwanie bez ponownego mnożenia macierzy
    while n > 0:
        ii = indx_i[0]
        mask[ii] = False

        # Usuwamy usunięty punkt ze spisu posortowanych odległości
        tmp = indx_i[indx_i != ii]
        indx_j = indx_j[indx_i != ii]
        indx_i = tmp.copy()

        tmp = indx_j[indx_j != ii]
        indx_i = indx_i[indx_j != ii]
        indx_j = tmp.copy()

        n -= 1

    # Przywrócenie punktów ekstremalnych
    for i in ideal:
        mask[i] = True

    return mask


def assigning_gens(nvars: int, nobjs: int) -> np.ndarray:
    """
    Generates a random base assignment of decision variables to objective players.
    Guarantees that every variable is assigned to at least one player to prevent
    'orphan' genes.

    Args:
        nvars (int): Number of decision variables.
        nobjs (int): Number of objective functions (players).

    Returns:
        np.ndarray: Boolean matrix of shape (nobjs, nvars) indicating assignment.
    """
    while True:
        if nvars <= nobjs:
            r = np.random.choice(range(nvars), size=(nobjs,), replace=True)
            # SECURITY CHECK: Ensure all unique variables are drawn at least once
            if len(np.unique(r)) == nvars:
                r2 = np.stack([r == i for i in range(nvars)])
                r2 = r2.T
                break
        else:
            r = np.random.randint(0, nobjs, size=(nvars,))
            r2 = np.stack([r == i for i in range(nobjs)])
            # Ensure no player owns ALL variables and no player owns ZERO variables
            if nvars >= nobjs and not np.any(np.all(r2, axis=1)) and not np.any(np.all(np.logical_not(r2), axis=1)):
                break
    return r2


def adaptive_linear_assigning_gens(front: np.ndarray, front_eval: np.ndarray, nvars: int, nobjs: int) -> np.ndarray:
    """
    Adaptive correlation-based allocation mechanism.

    Performs an inclusive hybrid variable assignment in three phases:
    1. Random Base Allocation: Guarantees full dimensional coverage.
    2. Statistical Influence Evaluation: Computes the Pearson correlation matrix
       between variables and objectives on the current Pareto front.
    3. Knowledge Injection: Uses a logical OR operator to assign highly correlated
       variables (hotspots) to players, overriding the exclusive random base.

    Args:
        front (np.ndarray): Current decision variable space of the Pareto front.
        front_eval (np.ndarray): Current objective space of the Pareto front.
        nvars (int): Number of decision variables.
        nobjs (int): Number of objective functions.

    Returns:
        np.ndarray: Boolean matrix of shape (nobjs, nvars) indicating the adaptive assignment.
    """
    # Phase 1: Random Base (Coverage Guarantee)
    patterns = assigning_gens(nvars, nobjs)

    # Fallback: If the front is too small for meaningful statistical analysis, return the random base
    if len(front) < 10:
        return patterns

    # Phase 2: Correlation Matrix Construction
    corr_matrix = np.zeros((nvars, nobjs))

    for i in range(nvars):
        var_data = front[:, i]
        # Avoid division by zero in correlation if standard deviation is near zero
        if np.std(var_data) > 1e-6:
            for j in range(nobjs):
                obj_data = front_eval[:, j]
                if np.std(obj_data) > 1e-6:
                    # Store the absolute value of the Pearson correlation coefficient
                    corr_matrix[i, j] = np.abs(np.corrcoef(var_data, obj_data)[0, 1])

    # Phase 3: Knowledge Injection (Inclusive Sharing of Significant Correlations)
    mean_corr = np.mean(corr_matrix)
    std_corr = np.std(corr_matrix)

    # Define a dynamic statistical threshold (μ + σ)
    dynamic_threshold = mean_corr + std_corr

    # Transpose matrix to match (nobjs, nvars) shape and create a boolean mask
    significant_mask = corr_matrix.T >= dynamic_threshold

    # Merge the random base with the statistical knowledge
    patterns = np.logical_or(patterns, significant_mask)

    return patterns


def adaptive_sparsity_gens(front, front_eval, nvars, nobjs):
    """
    Inteligentny przydział genów (Robin Hood DVA).
    Odbiera geny algorytmom z dobrymi wynikami i oddaje tym, które utknęły,
    wymuszając przeskakiwanie między lokalnymi minimami.
    """

    # Zabezpieczenie: jeśli front jest za mały do statystyki, zwracamy losowy przydział
    if len(front) < 20:
        return assigning_gens(nvars, nobjs)

    # --- ETAP 1: Obliczenie 'Potrzeby' (Need) dla każdego kryterium ---
    ideal = np.min(front_eval, axis=0)
    nadir = np.max(front_eval, axis=0)

    # Unikamy dzielenia przez zero (jeśli front zapadł się do jednego punktu)
    denom = nadir - ideal
    denom[denom < 1e-9] = 1e-9

    # Normalizacja wyników do przedziału [0, 1]
    F_norm = (front_eval - ideal) / denom

    # Wyznaczamy potrzebę.
    # mean_perf: im wyższa średnia, tym dalej populacja jest od minimum (gorzej).
    # spread: odchylenie standardowe - duże rozproszenie sugeruje brak konwergencji.
    mean_perf = np.mean(F_norm, axis=0)
    spread = np.std(F_norm, axis=0)

    # Wskaźnik Robin Hooda
    need = mean_perf + spread
    need_weights = need / (np.sum(need) + 1e-9)

    # --- ETAP 2: Obliczenie wpływu (korelacji) genów na kryteria ---
    C = np.zeros((nobjs, nvars))
    for i in range(nobjs):
        for j in range(nvars):
            std_x = np.std(front[:, j])
            std_f = np.std(front_eval[:, i])

            # Zabezpieczenie przed zerową wariancją (zmienna bezużyteczna w tym kroku)
            if std_x > 1e-6 and std_f > 1e-6:
                corr = np.abs(np.corrcoef(front[:, j], front_eval[:, i])[0, 1])
            else:
                corr = np.random.rand() * 0.01  # minimalny losowy szum
            C[i, j] = corr

    # --- ETAP 3: Przeciąganie liny (Ważenie Korelacji) ---
    # Gracz w potrzebie sztucznie zwiększa atrakcyjność genów dla siebie
    Weighted_C = C * need_weights[:, np.newaxis]

    # Przydział każdego genu do gracza, który zgłosił na niego "największe zapotrzebowanie"
    assignment = np.argmax(Weighted_C, axis=0)

    patterns = np.zeros((nobjs, nvars), dtype=bool)
    for j in range(nvars):
        patterns[assignment[j], j] = True

    # --- ETAP 4: Zabezpieczenie przed bezczynnością (Safeguard) ---
    # Każdy Gracz musi dostać co najmniej 1 gen, w przeciwnym razie wystąpi Deadlock!
    for i in range(nobjs):
        if np.sum(patterns[i, :]) == 0:
            # Znajdź najbogatszego Gracza (z największą liczbą przypisanych genów)
            richest = np.argmax(np.sum(patterns, axis=1))
            richest_genes = np.where(patterns[richest, :])[0]

            # Odbierz najbogatszemu gen, który ma na jego kryterium NAJMNIEJSZY wpływ
            if len(richest_genes) > 1:
                least_important_gene = richest_genes[np.argmin(C[richest, richest_genes])]
                patterns[richest, least_important_gene] = False
                patterns[i, least_important_gene] = True

    return patterns


def adaptive_shap_assigning_gens(front: np.ndarray, front_eval: np.ndarray, nvars: int, nobjs: int) -> np.ndarray:
    """
    Adaptive variable allocation mechanism driven by SHAP (Explainable AI).

    1. Random Base Allocation: Guarantees full dimensional coverage to prevent 'orphan' genes.
    2. Surrogate Modeling: Trains a fast Random Forest for each objective based on the current Pareto front.
    3. SHAP Knowledge Extraction: Uses TreeSHAP to calculate the non-linear, interaction-aware
       importance of every decision variable.
    4. Knowledge Injection: Assigns the most influential variables to their respective players.

    Args:
        front (np.ndarray): Current decision variable space of the Pareto front.
        front_eval (np.ndarray): Current objective space of the Pareto front.
        nvars (int): Number of decision variables.
        nobjs (int): Number of objective functions.

    Returns:
        np.ndarray: Boolean matrix of shape (nobjs, nvars) indicating the adaptive assignment.
    """
    # Random Base (Protection against 'orphan' genes)
    patterns = assigning_gens(nvars, nobjs)

    # SHAP requires a meaningful statistical sample to train the model.
    # If the archive has fewer than 20 points, return the random allocation.
    if len(front) < 20:
        return patterns

    shap_matrix = np.zeros((nvars, nobjs))

    # Surrogate Modeling and SHAP Explanation
    for j in range(nobjs):
        obj_data = front_eval[:, j]

        # Protection against degenerate data (zero variance)
        if np.std(obj_data) < 1e-6:
            continue

        # Train a fast Random Forest predicting the objective value based on genes (X)
        # model = RandomForestRegressor(n_estimators=20, max_depth=3, n_jobs=1)

        # Ultra-fast LightGBM setup tuned for Low-Latency surrogate modeling
        model = lgb.LGBMRegressor(
            n_estimators=20,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.1,
            n_jobs=1,
            verbose=-1,
            min_child_samples=5
        )

        model.fit(front, obj_data)

        # Use TreeExplainer, which is highly optimized and fast for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(front)

        # shap_values is a matrix of shape (num_points, nvars).
        # To get the 'global' feature importance, we take the mean of absolute values across all points.
        global_shap_importance = np.mean(np.abs(shap_values), axis=0)
        shap_matrix[:, j] = global_shap_importance

    # Knowledge Injection
    # Find the SHAP importance threshold for the entire matrix
    mean_shap = np.mean(shap_matrix)
    std_shap = np.std(shap_matrix)

    # Dynamic threshold (μ + σ)
    dynamic_threshold = mean_shap + std_shap

    # Transpose the matrix to match (nobjs, nvars) shape and create a boolean mask
    significant_mask = shap_matrix.T >= dynamic_threshold

    # Merge the significant SHAP insights with the random base (Logical OR)
    patterns = np.logical_or(patterns, significant_mask)

    return patterns

