import numpy as np
import random

# 初始化参数
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率
num_episodes = 1000  # 训练轮数

# 环境定义（简单的 5x5 网格）
num_states = 25  # 状态数量
num_actions = 4  # 上下左右
Q_table = np.zeros((num_states, num_actions))  # Q表初始化

# 定义状态转移和奖励
def step(state, action):
    # 示例：简单网格环境中的状态转移逻辑
    next_state = max(0, min(num_states-1, state + [1, -1, 5, -5][action]))  # 上下左右
    reward = 1 if next_state == num_states - 1 else -0.1  # 目标状态奖励
    return next_state, reward

# Q-Learning 训练
for episode in range(num_episodes):
    state = 0  # 初始状态
    while state != num_states - 1:
        # 选择动作（ε-贪心策略）
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(num_actions))  # 随机探索
        else:
            action = np.argmax(Q_table[state])  # 选择 Q 值最大的动作

        # 采取动作，获取奖励和下一个状态
        next_state, reward = step(state, action)

        # 更新 Q 值
        Q_table[state, action] = Q_table[state, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
        )

        state = next_state  # 进入下一个状态

# 输出训练后的 Q-table
print("训练完成的 Q-table:")
print(Q_table)
