## 1. [Analysis]
## 2. [EXTRA MECHANISMS]
## 3. [Code]

import numpy as np
import math

PARAMS = {
    'epsilon': 1e-09,
    'base_perfect_fit_bonus': 50.0,
    'complementary_bonus_scale': 0.3,
    'gap_discourage_power': 0.25,
    'size_preference_scale': 1e-06,
}

def _compute_instance_statistics(demands: np.ndarray, capacity: int) -> dict:
    """Compute meaningful instance characteristics for adaptive scoring."""
    mean_demand = demands.mean()
    std_demand = demands.std()
    cv_demand = std_demand / (mean_demand + 1e-6)
    
    # Expected number of bins if items randomly packed (lower bound)
    total_demand = demands.sum()
    expected_bins = total_demand / capacity
    
    return {
        'mean_demand': mean_demand,
        'cv_demand': cv_demand,
        'expected_bins': expected_bins,
        'total_demand': total_demand,
    }

def _compute_pairwise_scores(demands: np.ndarray, capacity: int, stats: dict) -> np.ndarray:
    """
    Compute all pairwise scores in one efficient pass.
    Returns: (residual_matrix, sum_matrix, size_diff_matrix)
    """
    n = len(demands)
    
    # Vectorized pairwise computations
    demands_i = demands[:, None]
    demands_j = demands[None, :]
    
    sum_matrix = demands_i + demands_j
    residual_matrix = capacity - sum_matrix
    size_diff_matrix = np.abs(demands_i - demands_j)
    
    return residual_matrix, sum_matrix, size_diff_matrix

def _compute_base_score(residual_matrix: np.ndarray) -> np.ndarray:
    """Core score: negative residual (higher fill = higher score)."""
    return -residual_matrix.astype(float)

def _apply_perfect_fit_bonus(score_matrix: np.ndarray, sum_matrix: np.ndarray, 
                           capacity: int, stats: dict) -> None:
    """Add adaptive bonus for pairs that sum exactly to capacity."""
    perfect_fit_mask = sum_matrix == capacity
    
    # Scale bonus by instance heterogeneity: more valuable when items are diverse
    cv = stats['cv_demand']
    bonus_scale = 1.0 + min(cv, 2.0)  # Cap at 3x for extreme heterogeneity
    bonus = PARAMS['base_perfect_fit_bonus'] * bonus_scale
    
    score_matrix[perfect_fit_mask] += bonus

def _apply_gap_discouragement(score_matrix: np.ndarray, residual_matrix: np.ndarray) -> None:
    """
    Continuously discourage all positive residuals.
    Smaller gaps receive stronger discouragement.
    """
    positive_gap_mask = residual_matrix > 0
    if not np.any(positive_gap_mask):
        return
    
    # Continuous discouragement: -residual^(power) where 0 < power < 1
    # This makes small gaps more penalized relative to their size
    gaps = residual_matrix[positive_gap_mask]
    penalty = -np.power(gaps, PARAMS['gap_discourage_power'])
    score_matrix[positive_gap_mask] += penalty

def _apply_complementary_bonus(score_matrix: np.ndarray, size_diff_matrix: np.ndarray,
                             sum_matrix: np.ndarray, capacity: int) -> None:
    """
    Reward complementary pairs (one large, one small) that efficiently use capacity.
    """
    # Pairs that use at least 70% of capacity and have complementary sizes
    good_fill_mask = sum_matrix > 0.7 * capacity
    complementary_mask = size_diff_matrix > 0.3 * capacity
    
    bonus_mask = good_fill_mask & complementary_mask
    if not np.any(bonus_mask):
        return
    
    # Bonus proportional to how complementary and well-filled the pair is
    fill_ratio = sum_matrix[bonus_mask] / capacity
    complement_ratio = size_diff_matrix[bonus_mask] / capacity
    bonus = PARAMS['complementary_bonus_scale'] * fill_ratio * complement_ratio * 100
    
    score_matrix[bonus_mask] += bonus

def _apply_size_ordering(score_matrix: np.ndarray, demands: np.ndarray) -> None:
    """
    Prefer larger items when scores are otherwise equal.
    This implements active size-based priority.
    """
    # Normalize demands to [0, 1] range for consistent scaling
    max_demand = demands.max()
    if max_demand > 0:
        normalized_sizes = demands / max_demand
        size_bias = normalized_sizes[None, :] * PARAMS['size_preference_scale']
        score_matrix += size_bias

def heuristics_v2(demands: np.ndarray, capacity: int) -> np.ndarray:
    """
    Generates an n x n heuristic matrix for BPP.
    heuristic_matrix[i, j] represents desirability of placing item j
    into a bin that currently contains item i.
    Higher value indicates higher desirability.
    """
    n = demands.shape[0]
    
    # 1. Compute instance statistics for adaptive behavior
    stats = _compute_instance_statistics(demands, capacity)
    
    # 2. Compute all pairwise matrices efficiently
    residual_matrix, sum_matrix, size_diff_matrix = _compute_pairwise_scores(
        demands, capacity, stats
    )
    
    # 3. Start with base score (minimize residual)
    score = _compute_base_score(residual_matrix)
    
    # 4. Reward perfect fits (adaptively scaled)
    _apply_perfect_fit_bonus(score, sum_matrix, capacity, stats)
    
    # 5. Continuously discourage all gaps (no arbitrary threshold)
    _apply_gap_discouragement(score, residual_matrix)
    
    # 6. Reward complementary pairs for balanced packing
    _apply_complementary_bonus(score, size_diff_matrix, sum_matrix, capacity)
    
    # 7. Ensure self-pairing is not allowed
    np.fill_diagonal(score, -np.inf)
    
    # 8. Apply size-based ordering for meaningful tie-breaking
    _apply_size_ordering(score, demands)
    
    # 9. Protect against zero/negative scores
    score = np.maximum(score, PARAMS['epsilon'])
    
    return score