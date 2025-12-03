# Mutation - Parameter Scan: Changed alpha blending factor from 0.7 to 0.9.
import numpy as np
import math

def _dynamic_radius_search(Positions, Best_pos, rg):
    direction = Best_pos - Positions
    random_step = (np.random.rand(*Positions.shape) - 0.5) * 2 * rg
    learning_rate = 0.3 + 0.2 * np.random.rand(Positions.shape[0], 1)
    return Positions + learning_rate * direction + random_step

def _preference_aware_repair(positions, Best_pos):
    alpha = 0.9
    repaired = alpha * Best_pos + (1 - alpha) * positions
    return np.clip(repaired, 0.0, 1.0)

def _fixed_diversity_injection(positions, rg):
    mask = np.random.rand(positions.shape[0]) < 0.2
    positions[mask] += (np.random.rand(*positions[mask].shape) - 0.5) * rg * 0.5
    return positions

def _conservative_gradient_mutation(positions, Best_pos, rg):
    if rg >= 0.3:
        return positions
    distances = np.linalg.norm(positions - Best_pos, axis=1)
    close_mask = distances < rg * 0.2
    mutation = (np.random.rand(*positions[close_mask].shape) - 0.5) * rg * 0.1
    positions[close_mask] += mutation
    return positions

def _elitist_preservation(positions, fitness):
    elite_count = max(1, int(positions.shape[0] * 0.1))
    elite_idx = np.argpartition(fitness, elite_count)[:elite_count]
    return positions[elite_idx], elite_idx

def heuristics_v2(Positions, Best_pos, Best_score, rg, stagnation_count=0, fitness=None):
    elite_pos, elite_idx = _elitist_preservation(Positions, fitness) if fitness is not None else (None, None)
    new_pos = _dynamic_radius_search(Positions, Best_pos, rg)
    new_pos = _fixed_diversity_injection(new_pos, rg)
    new_pos = _conservative_gradient_mutation(new_pos, Best_pos, rg)
    new_pos = _preference_aware_repair(new_pos, Best_pos)
    if elite_pos is not None:
        new_pos[elite_idx] = elite_pos
    return np.clip(new_pos, 0.0, 1.0)