import numpy as np
import math

def _normalize_distances(distance_matrix):
    """
    Normalize the distance matrix to mitigate scale effects.
    This provides a lightweight form of adaptation (Principle 6).
    Scaling by the mean preserves the relative structure while
    making the heuristic values less sensitive to absolute units.
    """
    mean_dist = np.mean(distance_matrix)
    # Avoid division by zero if all distances are zero (unlikely in TSP)
    if mean_dist < 1e-12:
        return distance_matrix
    return distance_matrix / mean_dist

def _compute_inverse_distance_matrix(norm_distance_matrix, epsilon=1e-12):
    """
    Core implementation of Strategy F01 (Inverse-Distance Heuristic Matrix).
    Encapsulates the Proximity-Intensity Mapping pillar.
    Strategy F02 (Epsilon Safeguard) is integrated here for robustness.
    """
    # Add epsilon to entire matrix to prevent division by zero
    protected_matrix = norm_distance_matrix + epsilon
    heuristic_matrix = 1.0 / protected_matrix
    # Explicitly set diagonal to zero (self-loops are never chosen)
    np.fill_diagonal(heuristic_matrix, 0.0)
    return heuristic_matrix

def _augment_with_nearest_neighbor_proxy(distance_matrix, base_heuristic, num_samples=5):
    """
    NEW STRATEGY: Nearest Neighbor Augmentation.
    Addresses the 'lacks global topology awareness' pitfall of pure inverse distance.
    Hypothesis: Edges that appear in cheap, greedy constructible tours (Nearest Neighbor)
    are more globally promising. This blends local proximity with a coarse global signal.
    """
    n = distance_matrix.shape[0]
    if n < 3 or num_samples < 1:
        return base_heuristic

    edge_frequency = np.zeros((n, n), dtype=np.float64)
    # Create a masked distance matrix to avoid selecting the same city
    masked_dist = distance_matrix.copy()
    np.fill_diagonal(masked_dist, np.inf)

    for _ in range(num_samples):
        start = np.random.randint(0, n)
        unvisited = list(range(n))
        current = start
        unvisited.remove(current)
        tour = [current]

        while unvisited:
            # Find nearest unvisited neighbor
            next_city = unvisited[np.argmin(masked_dist[current, unvisited])]
            # Record the undirected edge
            i, j = min(current, next_city), max(current, next_city)
            edge_frequency[i, j] += 1.0
            edge_frequency[j, i] += 1.0
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        # Close the tour (edge from last back to first)
        i, j = min(tour[-1], tour[0]), max(tour[-1], tour[0])
        edge_frequency[i, j] += 1.0
        edge_frequency[j, i] += 1.0

    # Normalize frequency to [0, 1] range and create a boost factor.
    # We use a softmax-inspired smooth boost: 1 + (frequency / max_freq)
    max_freq = edge_frequency.max()
    if max_freq > 0:
        boost_factor = 1.0 + 0.5 * (edge_frequency / max_freq)  # Max boost of 1.5x
    else:
        boost_factor = 1.0

    augmented_heuristic = base_heuristic * boost_factor
    return augmented_heuristic

def heuristics_v2(distance_matrix):
    """
    Orchestrator function for generating an advanced heuristic matrix for TSP.
    Manages the flow: input handling -> normalization -> core inverse calculation -> global augmentation.
    """
    # Ensure input is a NumPy array for consistent processing
    if hasattr(distance_matrix, 'cpu'):
        # Handle PyTorch Tensor
        dist_np = distance_matrix.cpu().numpy()
    elif hasattr(distance_matrix, 'numpy'):
        # Handle TensorFlow Tensor or similar
        dist_np = distance_matrix.numpy()
    else:
        # Assume it's a NumPy array or convertible
        dist_np = np.asarray(distance_matrix, dtype=np.float64)

    # 1. Normalize distances for scale robustness
    norm_dist = _normalize_distances(dist_np)

    # 2. Compute the foundational inverse-distance heuristic (Strategies F01 & F02)
    base_heuristic = _compute_inverse_distance_matrix(norm_dist)

    # 3. Augment with a global structure proxy (New Strategy)
    # Using a small number of samples for efficiency. This is a tunable parameter.
    final_heuristic = _augment_with_nearest_neighbor_proxy(
        dist_np, base_heuristic, num_samples=min(10, dist_np.shape[0] // 2)
    )

    # Final safeguard: ensure no infinite or NaN values
    final_heuristic = np.nan_to_num(final_heuristic, nan=0.0, posinf=0.0, neginf=0.0)
    # Ensure diagonal is explicitly zero
    np.fill_diagonal(final_heuristic, 0.0)

    return final_heuristic