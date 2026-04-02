# eval.py
# encoding: utf-8
import os
from aco import ACO
import numpy as np
import logging
import inspect

import gpt
from utils.utils import get_heuristic_name

possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)

N_ITERATIONS = 10
N_ANTS = 30


def solve(values, weights, capacities):
    # 计算启发式信息
    if len(inspect.getfullargspec(heuristics).args) == 3:
        heu = heuristics(values.copy(), weights.copy(), capacities.copy()) + 1e-9
    else:
        # 默认启发式
        heu = (values / np.maximum(np.sum(weights, axis=1), 1e-9)) + 1e-9
    
    heu[heu < 1e-9] = 1e-9
    
    aco = ACO(values, weights, capacities, heu, n_ants=N_ANTS)
    best_value = aco.run(N_ITERATIONS)
    return best_value


if __name__ == "__main__":
    # 固定参数：处理 train50_dataset.npy
    problem_size = 50  # 问题规模固定为50
    mood = 'train'  # 模式固定为训练集

    assert mood in ['train', 'val']

    basepath = os.path.dirname(__file__)
    # 若数据集不存在则自动生成
    if not os.path.isfile(os.path.join(basepath, "dataset/train50_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()

    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
        dataset = np.load(dataset_path)
        
        # 解析数据：价值、重量、容量
        values = dataset[:, :, 0]
        weights = dataset[:, :, 1:4]  # 3维重量
        capacities = dataset[:, 0, 4:7]  # 容量(每个实例相同)

        n_instances = values.shape[0]

        best_values = []
        for i in range(n_instances):
            val = solve(values[i], weights[i], capacities[i])
            best_values.append(val.item())

        # 输出平均最佳价值
        print(f"{np.mean(best_values):.2f}")