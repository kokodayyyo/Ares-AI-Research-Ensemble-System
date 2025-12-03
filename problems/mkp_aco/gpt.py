import numpy as np

def heuristics_v2(prize: np.ndarray, weight: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    norm_weights = _compute_normalized_weights(weight, capacity)
    ratios = _compute_geometric_mean_ratios(prize, norm_weights)
    capacity_factors = _compute_capacity_factors(weight, capacity)
    heuristic = _apply_adaptive_scaling(ratios, capacity_factors)
    heuristic = _add_exploration_noise(heuristic)
    return heuristic

def _compute_normalized_weights(weights: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    capacity = np.maximum(capacity, 1e-6)
    return weights / capacity[np.newaxis, :]

def _compute_geometric_mean_ratios(prize: np.ndarray, norm_weights: np.ndarray) -> np.ndarray:
    norm_weights = np.maximum(norm_weights, 1e-6)
    ratios = prize[:, np.newaxis] / norm_weights
    return np.exp(np.mean(np.log(ratios + 1e-6), axis=1))

def _compute_capacity_factors(weights: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    total_weight = np.sum(weights, axis=0)
    utilization = np.minimum(total_weight / (capacity + 1e-6), 1.0)
    return 1.0 - utilization

def _apply_adaptive_scaling(base_heuristic: np.ndarray, capacity_factors: np.ndarray) -> np.ndarray:
    weights = 1.0 / (1.0 - capacity_factors + 1e-6)
    avg_factor = np.sum(weights * capacity_factors) / np.sum(weights)
    return base_heuristic * (0.5 + 0.5 * avg_factor)

def _add_exploration_noise(heuristic: np.ndarray) -> np.ndarray:
    return heuristic * (1 + 1e-3 * np.random.randn(len(heuristic)))