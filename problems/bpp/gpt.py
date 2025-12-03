import numpy as np
import scipy.stats as stats

def quantum_tunneling_probability(energy_barrier, exploration_constant=0.1):
    """Calculate quantum tunneling probability based on energy barrier"""
    return np.exp(-energy_barrier / exploration_constant)

def meta_heuristic_fusion(skewness, residual_capacity, weights):
    """Dynamically blend heuristics based on problem characteristics"""
    skewness_factor = 0.5 + 0.5 * np.tanh(skewness)
    residual_factor = 1.0 / (residual_capacity + 1e-6)
    return weights * skewness_factor * residual_factor

def heuristics_v2(demands: np.ndarray, capacity: int) -> np.ndarray:
    n = demands.shape[0]
    epsilon = 1e-9
    exploration_constant = 0.1  # Tightened exploration bounds
    
    # Normalize demands and compute pairwise sums
    norm_demands = demands / capacity
    sum_matrix = np.add.outer(demands, demands) / capacity
    
    # Core heuristic scores
    residual = 1.0 - sum_matrix
    waste_score = np.where(sum_matrix <= 1.0, 1.0 - residual, epsilon)
    comp_score = np.exp(-((sum_matrix - 1.0)**2) / 0.1)
    residual_utility = 1.0 / (residual + epsilon)
    residual_utility[sum_matrix > 1.0] = epsilon
    
    # Quantum tunneling component with tighter bounds
    energy_barriers = np.abs(residual - residual.mean())
    tunneling_probs = quantum_tunneling_probability(energy_barriers, exploration_constant)
    quantum_boost = tunneling_probs * (1 + np.random.rand(n, n) * 0.05)  # Reduced randomness
    
    # Dynamic weighting based on demand distribution
    skewness = stats.skew(demands)
    weights = np.array([0.9, 0.85, 0.75])  # Updated weights based on Final Strategy Table
    dynamic_weights = meta_heuristic_fusion(skewness, residual.mean(), weights)
    
    # Combine scores with quantum boost
    combined = (dynamic_weights[0] * waste_score + 
               dynamic_weights[1] * comp_score + 
               dynamic_weights[2] * residual_utility) * quantum_boost
    
    # First-fit fallback with entropy-based activation (deprioritized)
    ff_score = np.tile(norm_demands, (n, 1))
    solution_entropy = stats.entropy(np.histogram(combined, bins=20)[0] + epsilon)
    adaptive_weight = 0.1 * (1 - np.exp(-solution_entropy))  # Reduced weight
    combined = (1 - adaptive_weight) * combined + adaptive_weight * ff_score
    
    # Final processing
    np.fill_diagonal(combined, 0)
    combined = np.maximum(combined, epsilon)
    
    # Normalize rows
    row_sums = combined.sum(axis=1, keepdims=True)
    heuristic_matrix = combined / (row_sums + epsilon)
    
    return heuristic_matrix