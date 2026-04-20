import numpy as np
import logging
from typing import Tuple

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
    z = x[:, np.newaxis] >= x
    z = np.all(z, axis=2)
    # An individual cannot dominate itself
    z[range(z.shape[0]), range(z.shape[0])] = False

    # Check if a solution is dominated by ANY other solution
    xx = np.any(z, axis=1)
    return np.logical_not(xx)


def get_not_dominated(populations_eval: np.ndarray) -> np.ndarray:
    """
    Filters a population to extract the non-dominated Pareto front.
    Utilizes a C-extension if available for performance, otherwise falls back to NumPy.

    Args:
        populations_eval (np.ndarray): Array of evaluated objective values.

    Returns:
        np.ndarray: A boolean mask representing non-dominated solutions.
    """
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


def front_suppression(front_eval: np.ndarray, front_max: int) -> np.ndarray:
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
    front_eval_norm = front_eval + np.abs(np.min(front_eval, axis=0))+1.0
    front_eval_norm = front_eval_norm/np.max(front_eval_norm, axis=0)

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
        tmp = indx_i[indx_i!=ii]
        indx_j = indx_j[indx_i!=ii]
        indx_i = tmp.copy()

        tmp = indx_j[indx_j!=ii]
        indx_i = indx_i[indx_j!=ii]
        indx_j = tmp.copy()

        n=n-1

    # Re-enable the boundary (ideal) points explicitly
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

