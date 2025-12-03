### Answer: Implementation According to Requirements
### Addressing Proposal Flaws
### Transformation of Strategies to Code

import numpy as np
import math
from scipy.spatial import ConvexHull

def heuristics_v2(distance_matrix):
    """
    Enhanced TSP heuristic combining local refinement, strategic perturbation,
    edge frequency analysis, and adaptive cooling.
    """
    # Convert to numpy if input is torch tensor
    if hasattr(distance_matrix, 'numpy'):
        distance_matrix = distance_matrix.numpy()
    
    # Initialize parameters
    num_cities = distance_matrix.shape[0]
    max_iter = min(1000, num_cities * 10)
    candidate_pool_size = min(10, num_cities // 2)
    
    # Generate initial solutions
    initial_solutions = []
    initial_solutions.append(_nearest_neighbor_init(distance_matrix))
    initial_solutions.append(_convex_hull_init(distance_matrix))
    
    # Initialize candidate pool
    candidate_pool = _manage_candidate_pool(initial_solutions, distance_matrix, 
                                          pool_size=candidate_pool_size)
    
    # Initialize edge frequency matrix
    edge_freq = _compute_edge_frequency_matrix(num_cities, candidate_pool)
    
    best_solution = None
    best_cost = float('inf')
    
    for iteration in range(max_iter):
        # Select solution from pool
        current_sol = candidate_pool[np.random.randint(len(candidate_pool))]
        
        # Apply local refinement
        refined_sol = _apply_2opt_refinement(current_sol, distance_matrix)
        refined_cost = _compute_tour_cost(refined_sol, distance_matrix)
        
        # Apply perturbation with decreasing probability
        if np.random.rand() < 0.5 * (1 - iteration/max_iter):
            perturbed_sol = _apply_double_bridge_perturbation(refined_sol)
            perturbed_cost = _compute_tour_cost(perturbed_sol, distance_matrix)
            
            # Adaptive acceptance
            if _adaptive_cooling_schedule(perturbed_cost, refined_cost, iteration, max_iter):
                refined_sol = perturbed_sol
                refined_cost = perturbed_cost
        
        # Update candidate pool
        candidate_pool = _manage_candidate_pool(candidate_pool + [refined_sol], 
                                              distance_matrix, 
                                              pool_size=candidate_pool_size)
        
        # Update edge frequencies
        edge_freq = _compute_edge_frequency_matrix(num_cities, candidate_pool)
        
        # Track best solution
        if refined_cost < best_cost:
            best_solution = refined_sol
            best_cost = refined_cost
    
    # Convert best solution to heuristic matrix
    heuristic_matrix = _solution_to_heuristic(best_solution, distance_matrix, edge_freq)
    return heuristic_matrix

def _nearest_neighbor_init(distance_matrix):
    """Generate initial solution using nearest neighbor heuristic."""
    num_cities = distance_matrix.shape[0]
    unvisited = set(range(num_cities))
    tour = [np.random.choice(num_cities)]
    unvisited.remove(tour[0])
    
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda city: distance_matrix[last, city])
        tour.append(next_city)
        unvisited.remove(next_city)
    
    return tour

def _convex_hull_init(distance_matrix):
    """Generate initial solution using convex hull as skeleton."""
    points = np.random.rand(distance_matrix.shape[0], 2)  # Mock coordinates
    hull = ConvexHull(points)
    hull_points = list(hull.vertices)
    remaining = [i for i in range(distance_matrix.shape[0]) if i not in hull_points]
    
    # Insert remaining points nearest to hull edges
    for point in remaining:
        min_dist = float('inf')
        insert_pos = -1
        for i in range(len(hull_points)):
            j = (i + 1) % len(hull_points)
            dist = distance_matrix[hull_points[i], point] + distance_matrix[point, hull_points[j]]
            if dist < min_dist:
                min_dist = dist
                insert_pos = i + 1
        hull_points.insert(insert_pos, point)
    
    return hull_points

def _apply_2opt_refinement(tour, distance_matrix):
    """Apply 2-opt local optimization to improve the tour."""
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 2, len(tour)):
                a, b = tour[i], tour[i+1]
                c, d = tour[j], tour[(j+1)%len(tour)]
                
                # Check if swap would improve the tour
                if distance_matrix[a,b] + distance_matrix[c,d] > distance_matrix[a,c] + distance_matrix[b,d]:
                    tour[i+1:j+1] = tour[j:i:-1]  # Reverse the segment
                    improved = True
    return tour

def _apply_double_bridge_perturbation(tour):
    """Apply double-bridge move for strategic perturbation."""
    if len(tour) < 8:
        return tour.copy()
    
    # Split tour into 4 segments
    split_points = sorted(np.random.choice(len(tour), 4, replace=False))
    a, b, c, d = split_points
    
    # Reconnect segments in different order: 1-3-2-4
    new_tour = tour[:a+1] + tour[c:d+1] + tour[b+1:c] + tour[a+1:b+1] + tour[d+1:]
    return new_tour

def _compute_edge_frequency_matrix(num_cities, solutions):
    """Compute edge frequency matrix from candidate solutions."""
    freq_matrix = np.zeros((num_cities, num_cities))
    
    for solution in solutions:
        for i in range(len(solution)):
            city1 = solution[i]
            city2 = solution[(i+1)%len(solution)]
            freq_matrix[city1, city2] += 1
            freq_matrix[city2, city1] += 1
    
    # Normalize and add small constant
    if len(solutions) > 0:
        freq_matrix = freq_matrix / len(solutions)
    return freq_matrix + 1e-6

def _adaptive_cooling_schedule(new_cost, current_cost, iteration, max_iter):
    """Determine whether to accept worse solution based on adaptive cooling."""
    temperature = max(0.1, 1.0 - (iteration / max_iter))
    if new_cost < current_cost:
        return True
    else:
        delta = new_cost - current_cost
        probability = math.exp(-delta / (temperature * current_cost))
        return np.random.rand() < probability

def _manage_candidate_pool(solutions, distance_matrix, pool_size=5):
    """Maintain pool of diverse high-quality solutions."""
    # Score solutions
    scored = [(sol, _compute_tour_cost(sol, distance_matrix)) for sol in solutions]
    
    # Sort by quality
    scored.sort(key=lambda x: x[1])
    
    # Select top solutions with some diversity
    pool = []
    for sol, cost in scored:
        if len(pool) >= pool_size:
            break
        # Check diversity
        if not pool or _solution_diversity(sol, pool) > 0.2:
            pool.append(sol)
    
    # If pool not full, add random solutions
    while len(pool) < pool_size and len(solutions) > len(pool):
        pool.append(np.random.permutation(len(distance_matrix)).tolist())
    
    return pool

def _solution_diversity(solution, pool):
    """Calculate minimum edge overlap between solution and pool."""
    if not pool:
        return 1.0
    
    solution_edges = set(zip(solution, solution[1:] + solution[:1]))
    min_overlap = 1.0
    for other in pool:
        other_edges = set(zip(other, other[1:] + other[:1]))
        overlap = len(solution_edges & other_edges) / len(solution_edges)
        min_overlap = min(min_overlap, overlap)
    
    return 1 - min_overlap

def _compute_tour_cost(tour, distance_matrix):
    """Compute total cost of a tour."""
    cost = 0
    for i in range(len(tour)):
        cost += distance_matrix[tour[i], tour[(i+1)%len(tour)]]
    return cost

def _solution_to_heuristic(solution, distance_matrix, edge_freq):
    """Convert best solution to heuristic matrix."""
    epsilon = 1e-15
    heuristic = np.zeros_like(distance_matrix)
    
    # Base heuristic is inverse distance
    heuristic = 1.0 / (distance_matrix + epsilon)
    
    # Boost edges that appear in good solutions
    for i in range(len(solution)):
        city1 = solution[i]
        city2 = solution[(i+1)%len(solution)]
        heuristic[city1, city2] *= (1 + edge_freq[city1, city2])
        heuristic[city2, city1] *= (1 + edge_freq[city1, city2])
    
    return heuristic