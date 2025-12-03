# utils/utils.py
import inspect

def get_heuristic_name(module, possible_func_names):
    """获取启发式函数的名称"""
    for name in possible_func_names:
        if hasattr(module, name):
            return name
    return "heuristics_v2"  # 默认启发式函数

def parse_mkp_instance(instance):
    """解析MKP实例"""
    values = instance[:, 0]
    weights = instance[:, 1:4]  # 3维重量
    capacities = instance[0, 4:7]  # 容量
    return values, weights, capacities

def check_solution_feasibility(solution, weights, capacities):
    """检查解的可行性"""
    for d in range(weights.shape[1]):
        total_weight = np.sum(weights[:, d] * solution)
        if total_weight > capacities[d]:
            return False, total_weight - capacities[d]
    return True, 0