# gen_inst.py
import os
import numpy as np

# 多维背包问题参数
NUM_DIMENSIONS = 3  # 3维背包
VALUE_LOW = 10
VALUE_HIGH = 100
WEIGHT_LOW = 1
WEIGHT_HIGH = 50
CAPACITY_RATIO = 0.4  # 容量与总重量的比例

def gen_instance(n_items):
    """生成一个MKP实例"""
    # 物品价值
    values = np.random.randint(low=VALUE_LOW, high=VALUE_HIGH+1, size=n_items)
    
    # 物品重量(多维)
    weights = np.random.randint(
        low=WEIGHT_LOW, 
        high=WEIGHT_HIGH+1, 
        size=(n_items, NUM_DIMENSIONS)
    )
    
    # 计算总重量并设置背包容量
    total_weights = np.sum(weights, axis=0)
    capacities = (total_weights * CAPACITY_RATIO).astype(int)
    
    # 确保容量为正数
    capacities = np.maximum(capacities, 1)
    
    return np.concatenate([
        values.reshape(-1, 1),
        weights,
        capacities.reshape(1, -1).repeat(n_items, axis=0)
    ], axis=1)

def generate_datasets():
    basepath = os.path.dirname(__file__)
    os.makedirs(os.path.join(basepath, "dataset"), exist_ok=True)
    
    np.random.seed(1234)
    
    # 训练集
    for problem_size in [50]:
        n_instances = 10
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(os.path.join(basepath, f'dataset/train{problem_size}_dataset.npy'), dataset)

    # 验证集和测试集
    for problem_size in [20, 50, 100]:
        for mood in ['val', 'test']:
            n_instances = 64
            dataset = []
            for i in range(n_instances):
                inst = gen_instance(problem_size)
                dataset.append(inst)
            dataset = np.array(dataset)
            np.save(os.path.join(basepath, f'dataset/{mood}{problem_size}_dataset.npy'), dataset)

if __name__ == "__main__":
    generate_datasets()