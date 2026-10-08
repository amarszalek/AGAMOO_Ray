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


def get_not_dominated(populations_eval: np.ndarray, epsilon: float = 0.0, x=None, x_bounds=None, eps_x=None,
                      alpha_relative=False) -> np.ndarray:
    """
    Filters a population to extract the non-dominated Pareto front.
    Utilizes a C-extension if available for performance, otherwise falls back to NumPy.
    - If epsilon > 0.0: Applies Grid Epsilon-Dominance (Thins the front).
    - If epsilon < 0.0: Applies Alpha Dominance (Thickens the front, keeps slightly dominated solutions).
    - If epsilon == 0.0: Classic Pareto Dominance.

    Args:
        populations_eval (np.ndarray): Array of evaluated objective values.
        epsilon (float): Epsilon-Dominance threshold.

    Returns:
        np.ndarray: A boolean mask representing non-dominated solutions.
    """

    if epsilon > 0.0:
        boxes = np.floor(populations_eval / epsilon)
        worse_or_equal = np.all(boxes[:, np.newaxis] >= boxes, axis=2)
        strictly_worse = np.any(boxes[:, np.newaxis] > boxes, axis=2)
        valid_indices = np.where(~np.any(worse_or_equal & strictly_worse, axis=1))[0]

        xcell = None
        if x is not None and eps_x is not None:
            lo, hi = (None, None) if x_bounds is None else (x_bounds[:, 0], x_bounds[:, 1])
            xcell = np.floor(_minmax(x, lo, hi) / eps_x).astype(np.int64)

        unique_boxes = {}
        for idx in valid_indices:
            key = tuple(boxes[idx]) if xcell is None else (tuple(boxes[idx]), tuple(xcell[idx]))
            dist = np.sum(populations_eval[idx])
            if key not in unique_boxes or dist < unique_boxes[key][1]:
                unique_boxes[key] = (idx, dist)

        final_mask = np.zeros(populations_eval.shape[0], dtype=bool)
        for idx, _ in unique_boxes.values():
            final_mask[idx] = True
        return final_mask

    elif epsilon < 0.0:
        tol = abs(epsilon)
        if alpha_relative:
            span = populations_eval.max(axis=0) - populations_eval.min(axis=0)
            tol = tol * np.where(span > 1e-12, span, 1.0)
        is_significantly_worse = np.all(populations_eval[:, np.newaxis] >= populations_eval + tol, axis=2)
        return ~np.any(is_significantly_worse, axis=1)

    else:
        # --- Classic Pareto Dominance ---
        if CEXT:
            mask = np.zeros(populations_eval.shape[0], dtype=np.int32)
            cutils.cget_not_dominated(populations_eval, mask)
            mask = mask.astype(bool)
        else:
            mask = pairwise_dominance(populations_eval)
        return mask


def pairwise_distance(x: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean pairwise distance matrix for crowding distance estimation.
    """
    return np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)


def front_suppression(front: np.ndarray, front_eval: np.ndarray, front_max: int, mode: str = 'objectives', x_bounds=None) -> np.ndarray:
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
        return _dual_omni_suppression(front, front_eval, front_max, x_bounds=x_bounds)
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
    protected = set(int(i) for i in ideal)
    while n > 0 and len(indx_i) > 0:
        ii = indx_i[0]
        if ii in protected:
            ii = indx_j[0]
        if ii in protected:  # both ends of the closest pair are protected
            indx_i, indx_j = indx_i[1:], indx_j[1:]
            continue
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


def _minmax(a, lo=None, hi=None):
    """Min-max do [0, 1]; lo/hi = granice problemu albo None (zakres bieżącego zbioru)."""
    lo = a.min(axis=0) if lo is None else np.asarray(lo, dtype=float)
    hi = a.max(axis=0) if hi is None else np.asarray(hi, dtype=float)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (a - lo) / span

def _dual_omni_suppression(front: np.ndarray, front_eval: np.ndarray, front_max: int, x_bounds=None) -> np.ndarray:
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
    n_pts = front_eval.shape[0]
    if n_pts <= front_max:
        return np.ones(n_pts, dtype=bool)

    lo, hi = (None, None) if x_bounds is None else (x_bounds[:, 0], x_bounds[:, 1])
    Xn = _minmax(front, lo, hi) / np.sqrt(front.shape[1])
    Fn = _minmax(front_eval) / np.sqrt(front_eval.shape[1])
    D = np.maximum(pairwise_distance(Fn), pairwise_distance(Xn))
    np.fill_diagonal(D, np.inf)

    keep = np.ones(n_pts, dtype=bool)
    protected = np.zeros(n_pts, dtype=bool)
    protected[np.argmin(front_eval, axis=0)] = True

    for _ in range(n_pts - front_max):
        cand = np.where(keep & ~protected)[0]
        if cand.size == 0:
            break
        Ds = np.sort(D[np.ix_(cand, np.where(keep)[0])], axis=1)
        k = min(2, Ds.shape[1])
        order = np.lexsort(tuple(Ds[:, i] for i in reversed(range(k))))
        victim = cand[order[0]]
        keep[victim] = False
        D[victim, :] = np.inf
        D[:, victim] = np.inf
    return keep


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


def correlation_ratio(front, front_eval, n_bins=10):
    n, nvars = front.shape
    nobjs = front_eval.shape[1]
    b = max(2, min(n_bins, n // 5))
    fz = front_eval - front_eval.mean(axis=0)
    tot = np.sum(fz ** 2, axis=0)
    eta = np.zeros((nvars, nobjs))
    for i in range(nvars):
        edges = np.quantile(front[:, i], np.linspace(0, 1, b + 1)[1:-1])
        bins = np.searchsorted(edges, front[:, i])
        cnt = np.bincount(bins, minlength=b).astype(float)
        nz = cnt > 0
        for j in range(nobjs):
            s = np.bincount(bins, weights=fz[:, j], minlength=b)
            eta[i, j] = np.sum(s[nz] ** 2 / cnt[nz]) / tot[j] if tot[j] > 1e-12 else 0.0
    return eta


def adaptive_eta_assigning_gens(front, front_eval, nvars, nobjs):
    patterns = assigning_gens(nvars, nobjs)
    if len(front) < 20:
        return patterns
    eta = correlation_ratio(front, front_eval)
    return np.logical_or(patterns, eta.T >= eta.mean() + eta.std())


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
    Intelligent gene allocation (Robin Hood DVA).
    Takes genes from algorithms with good performance and gives them to those
    that are stuck, forcing jumps between local minima.
    """

    # Safeguard: if the front is too small for statistics, return a random assignment
    if len(front) < 20:
        return assigning_gens(nvars, nobjs)

    # --- PHASE 1: Calculate 'Need' for each objective ---
    ideal = np.min(front_eval, axis=0)
    nadir = np.max(front_eval, axis=0)

    # Avoid division by zero (if the front collapsed to a single point)
    denom = nadir - ideal
    denom[denom < 1e-9] = 1e-9

    # # Normalize results to [0, 1] interval
    F_norm = (front_eval - ideal) / denom

    # Determine the need
    # mean_perf: the higher the average, the further the population is from the minimum (worse).
    # spread: standard deviation - large dispersion suggests a lack of convergence.
    mean_perf = np.mean(F_norm, axis=0)
    spread = np.std(F_norm, axis=0)

    # Robin Hood Indicator
    need = mean_perf + spread
    need_weights = need / (np.sum(need) + 1e-9)

    # --- PHASE 2: Calculate the influence (correlation) of genes on criteria ---
    C = np.zeros((nobjs, nvars))
    for i in range(nobjs):
        for j in range(nvars):
            std_x = np.std(front[:, j])
            std_f = np.std(front_eval[:, i])

            # Safeguard against zero variance (useless variable in this step)
            if std_x > 1e-6 and std_f > 1e-6:
                corr = np.abs(np.corrcoef(front[:, j], front_eval[:, i])[0, 1])
            else:
                corr = np.random.rand() * 0.01  # minimal random noise
            C[i, j] = corr

    # --- PHASE 3: Tug of War (Correlation Weighting) ---
    # A player in need artificially increases the attractiveness of genes for themselves
    Weighted_C = C * need_weights[:, np.newaxis]

    # Assign each gene to the player who expressed the "greatest demand" for it
    assignment = np.argmax(Weighted_C, axis=0)
    patterns = np.zeros((nobjs, nvars), dtype=bool)
    for j in range(nvars):
        patterns[assignment[j], j] = True

    # --- PHASE 4: Safeguard against inactivity (Deadlock Prevention) ---
    # Every Player must receive at least 1 gene, otherwise a Deadlock will occur!
    for i in range(nobjs):
        if np.sum(patterns[i, :]) == 0:
            # Find the wealthiest Player (with the highest number of assigned genes)
            richest = np.argmax(np.sum(patterns, axis=1))
            richest_genes = np.where(patterns[richest, :])[0]

            # Take a gene from the wealthiest player that has the LEAST impact on their criterion
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

