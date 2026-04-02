import numpy as np
import math

def heuristics_v2(Positions: np.ndarray, Best_pos: np.ndarray, Best_score: float, rg: float) -> np.ndarray:
    """
    Enhanced position update with optimized learning balance, refined constraint handling,
    and improved boundary management for superior WSN optimization performance.
    """
    SearchAgents_no, dim = Positions.shape
    new_positions = np.zeros_like(Positions)
    
    # F01: Global Best-Guided Exploitation with empirically optimal 0.7/0.3 balance
    direction_to_best = Best_pos - Positions
    
    # F03: Dynamic step size control - apply rg consistently to all components
    # Research directive: Refine rg parameter scheduling for better exploration-exploitation balance
    exploration_component = 0.7 * rg
    exploitation_component = 0.3 * (1 - np.clip(rg, 0.1, 1.0))  # Ensure minimum exploration
    
    # I02: Numerical stability - protect against division by zero
    epsilon = 1e-10
    base_learning_rate = exploration_component + exploitation_component + epsilon
    
    # Adaptive learning with controlled randomness
    learning_rate = base_learning_rate * (0.8 + 0.2 * np.random.rand(SearchAgents_no, 1))
    
    # F04: Feasibility-aware objective with clean threshold logic
    # F02: Constraint-violation reduction with simple threshold-based approach
    if Best_score > 1000:
        # In infeasible region: stronger perturbation to escape constraint violations
        constraint_force = np.random.randn(SearchAgents_no, dim) * 0.6 * rg
    else:
        # In feasible region: finer perturbation for local refinement
        constraint_force = np.random.randn(SearchAgents_no, dim) * 0.15 * rg
    
    # U01: Controlled random perturbation for diversity (secondary to main exploration)
    random_perturbation = (np.random.rand(SearchAgents_no, dim) - 0.5) * 2 * rg
    
    # Integrated position update with all rg-scaled components
    new_positions = Positions + (
        learning_rate * direction_to_best +
        constraint_force +
        random_perturbation
    )
    
    # F05: Enhanced boundary handling with reflection for coordinates, clipping for power
    new_positions = apply_wsn_specific_boundary_handling(new_positions, dim)
    
    # U02: Elitism - preserve the global best solution
    best_agent_idx = np.argmin(np.linalg.norm(Positions - Best_pos, axis=1))
    new_positions[best_agent_idx] = Best_pos
    
    return new_positions

def apply_wsn_specific_boundary_handling(positions: np.ndarray, dim: int) -> np.ndarray:
    """
    WSN-specific boundary handling: reflective for coordinates (X, Y) and 
    clipping for power values to maintain physical constraints.
    """
    result = positions.copy()
    
    for i in range(0, dim, 3):
        # X coordinates: reflective boundary [0.0, 50.0]
        result[:, i] = reflective_boundary_with_clamp(positions[:, i], 0.0, 50.0)
        
        # Y coordinates: reflective boundary [0.0, 50.0]
        result[:, i+1] = reflective_boundary_with_clamp(positions[:, i+1], 0.0, 50.0)
        
        # Power values: hard clipping [0.0, 30.0]
        result[:, i+2] = np.clip(positions[:, i+2], 0.0, 30.0)
    
    return result

def reflective_boundary_with_clamp(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """
    Apply reflective boundary handling with final clamping to ensure values
    stay within bounds after reflection.
    """
    result = values.copy()
    range_width = upper - lower
    
    # Handle values below lower bound with reflection
    below_mask = values < lower
    reflection_count = np.zeros_like(values)
    
    # Multiple reflections might be needed for extreme values
    while np.any(below_mask):
        result[below_mask] = 2 * lower - result[below_mask]
        reflection_count[below_mask] += 1
        below_mask = result < lower
        
        # Safety: clamp after too many reflections to prevent infinite loop
        too_many_reflections = reflection_count > 5
        if np.any(too_many_reflections):
            result[too_many_reflections] = lower
            break
    
    # Handle values above upper bound with reflection
    above_mask = values > upper
    reflection_count = np.zeros_like(values)
    
    while np.any(above_mask):
        result[above_mask] = 2 * upper - result[above_mask]
        reflection_count[above_mask] += 1
        above_mask = result > upper
        
        # Safety: clamp after too many reflections
        too_many_reflections = reflection_count > 5
        if np.any(too_many_reflections):
            result[too_many_reflections] = upper
            break
    
    # Final clamping to ensure all values are within bounds
    return np.clip(result, lower, upper)