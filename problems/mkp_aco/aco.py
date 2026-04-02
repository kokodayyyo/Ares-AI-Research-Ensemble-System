# encoding: utf-8
# aco.py (完全修复版本)
import torch
import numpy as np


class ACO():
    def __init__(self,
                 values,
                 weights,
                 capacities,
                 heuristic,
                 n_ants=30,  # 增加蚂蚁数量
                 decay=0.95,  # 提高衰减率
                 alpha=2,  # 提高信息素重要性
                 beta=3,  # 提高启发式信息重要性
                 device='cpu',
                 ):

        self.n_items = len(values)
        self.n_dimensions = weights.shape[1]

        # 确保所有数据都是Tensor
        self.values = torch.tensor(values, device=device, dtype=torch.float32)
        self.weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self.capacities = torch.tensor(capacities, device=device, dtype=torch.float32)
        self.heuristic = torch.tensor(heuristic, device=device, dtype=torch.float32)

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        # 初始化信息素为启发式值的函数
        self.pheromone = torch.ones(self.n_items, device=device, dtype=torch.float32) * 10.0

        self.best_solution = None
        self.best_value = -float('inf')
        self.device = device

    @torch.no_grad()
    def run(self, n_iterations):
        for iteration in range(n_iterations):
            # 生成解决方案
            solutions, values = self.gen_solutions()

            # 更新最优解
            best_iter_value, best_iter_idx = values.max(dim=0)
            if best_iter_value > self.best_value:
                self.best_value = best_iter_value
                self.best_solution = solutions[:, best_iter_idx].clone()

            # 更新信息素（传递当前迭代的解）
            self.update_pheromone(solutions, values)

            # 可选：打印进度
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best Value = {self.best_value:.2f}")

        return self.best_value

    @torch.no_grad()
    def update_pheromone(self, solutions, values):
        """根据当前迭代的解更新信息素"""
        # 信息素挥发
        self.pheromone = self.pheromone * self.decay

        # 只更新有效解的信息素
        valid_mask = values > 0
        if valid_mask.any():
            # 最佳解的信息素更新更强
            best_value, best_idx = values.max(dim=0)
            best_solution = solutions[:, best_idx]
            self.pheromone[best_solution == 1] += best_value * 2.0

            # 其他有效解也更新，但权重较低
            for i in range(self.n_ants):
                if valid_mask[i] and i != best_idx:
                    solution = solutions[:, i]
                    value = values[i]
                    self.pheromone[solution == 1] += value * 0.5

        # 信息素边界限制
        self.pheromone.clamp_(min=0.1, max=1e6)

    @torch.no_grad()
    def gen_solutions(self):
        """为每只蚂蚁构建解决方案"""
        solutions = torch.zeros((self.n_items, self.n_ants), dtype=torch.float32, device=self.device)
        values = torch.zeros(self.n_ants, device=self.device)

        for ant in range(self.n_ants):
            solution = self.construct_solution_for_ant()
            solutions[:, ant] = solution
            values[ant] = torch.sum(self.values * solution)

        return solutions, values

    @torch.no_grad()
    def construct_solution_for_ant(self):
        """为单只蚂蚁构建解决方案"""
        solution = torch.zeros(self.n_items, device=self.device)
        remaining_capacities = self.capacities.clone()

        # 创建可用物品列表
        available_items = torch.ones(self.n_items, dtype=torch.bool, device=self.device)

        while available_items.any():
            # 找出所有可行的物品
            feasible_mask = available_items.clone()
            for d in range(self.n_dimensions):
                feasible_mask &= (self.weights[:, d] <= remaining_capacities[d])

            if not feasible_mask.any():
                break  # 没有可行物品

            feasible_indices = torch.where(feasible_mask)[0]

            # 计算选择概率
            pheromone = self.pheromone[feasible_indices] ** self.alpha
            heuristic = self.heuristic[feasible_indices] ** self.beta
            probabilities = pheromone * heuristic

            if probabilities.sum() == 0:
                probabilities = torch.ones_like(probabilities)

            probabilities /= probabilities.sum()

            # 轮盘赌选择
            selected_idx = torch.multinomial(probabilities, 1).item()
            item_idx = feasible_indices[selected_idx]

            # 添加到解决方案
            solution[item_idx] = 1
            remaining_capacities -= self.weights[item_idx]
            available_items[item_idx] = False

        return solution