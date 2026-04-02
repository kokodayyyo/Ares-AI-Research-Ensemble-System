# Mutation - Parameter Scan: Changed capacity threshold from 0.8 to 0.9 in _depot_centric_clustering
import numpy as np
import math
import scipy
import torch

def heuristics_v2(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    heuristic = 1 / (distance_matrix + 1e-6)
    heuristic *= _demand_aware_proximity(demands, distance_matrix)
    heuristic *= _depot_centric_clustering(coordinates, demands, capacity)
    heuristic *= _edge_saturation_score(distance_matrix, coordinates)
    heuristic *= _normalized_savings(distance_matrix, demands, capacity)
    heuristic *= _inter_route_2opt(distance_matrix, demands, capacity)
    return heuristic / (np.max(heuristic) + 1e-6)

def _demand_aware_proximity(demands: np.ndarray, distance_matrix: np.ndarray) -> np.ndarray:
    demand_factor = np.outer(demands, demands)
    distance_factor = 1 / (distance_matrix + 1e-6)
    return np.sqrt(demand_factor) * distance_factor

def _depot_centric_clustering(coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    n = len(demands)
    cluster_score = np.ones((n, n))
    depot_coord = coordinates[0]
    depot_dists = np.linalg.norm(coordinates - depot_coord, axis=1)
    demand_sums = demands[:, None] + demands[None, :]
    
    for i in range(1, n):
        for j in range(1, n):
            if demand_sums[i, j] > capacity * 0.9:
                cluster_score[i, j] *= 0.6
            if max(depot_dists[i], depot_dists[j]) > np.percentile(depot_dists, 75):
                cluster_score[i, j] *= 0.7
    
    median_dist = np.median(depot_dists)
    for i in range(1, n):
        if depot_dists[i] > median_dist:
            cluster_score[i, 0] = cluster_score[0, i] = 1.5
    
    return cluster_score

def _edge_saturation_score(distance_matrix: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    n = len(coordinates)
    saturation = np.ones((n, n))
    depot_dists = np.linalg.norm(coordinates - coordinates[0], axis=1)
    median_dist = np.median(depot_dists)
    
    for i in range(n):
        boost = 1.5 + 1.0 / (1 + np.exp(-0.5*(depot_dists[i] - median_dist)))
        saturation[i, 0] = saturation[0, i] = boost
    
    mean_dist = np.mean(distance_matrix[1:, 1:])
    for i in range(1, n):
        for j in range(1, n):
            if distance_matrix[i, j] < mean_dist * 0.5:
                saturation[i, j] *= 1.3
    return saturation

def _normalized_savings(distance_matrix: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    n = len(demands)
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                savings[i, j] = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]
    
    max_saving = np.max(savings) + 1e-6
    demand_proxy = (demands[:, None] + demands[None, :]) / (2 * capacity)
    return (savings / max_saving) * (1.2 - demand_proxy)

def _inter_route_2opt(distance_matrix: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    return np.ones_like(distance_matrix)