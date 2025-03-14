import math
import random

# 目标函数
def f(x):
    return x**2

# 模拟退火算法
def simulated_annealing():
    # 初始解
    x_current = 8
    f_current = f(x_current)
    
    # 初始温度
    T = 1000
    # 降温系数
    alpha = 0.95
    # 最小温度
    T_min = 0.01
    
    while T > T_min:
        # 生成新解
        x_new = x_current + random.uniform(-1, 1)
        f_new = f(x_new)
        
        # 计算接受概率
        delta_f = f_new - f_current
        if delta_f < 0 or random.random() < math.exp(-delta_f / T):
            x_current = x_new
            f_current = f_new
        
        # 降温
        T = T * alpha
    
    return x_current, f_current

# 运行算法
x_opt, f_opt = simulated_annealing()
print(f"最优解: x = {x_opt}, f(x) = {f_opt}")