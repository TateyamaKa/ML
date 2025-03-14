import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
n_samples = 1000

# 两个高斯分布的参数
mu1, sigma1 = 2, 0.5
mu2, sigma2 = -1, 1

# 生成数据
data1 = np.random.normal(mu1, sigma1, n_samples // 2)
data2 = np.random.normal(mu2, sigma2, n_samples // 2)
data = np.concatenate([data1, data2])

# 可视化数据
plt.hist(data, bins=50, density=True)
plt.title("Generated Data")
plt.show()
# 初始化参数
mu1_init, sigma1_init = 1, 1
mu2_init, sigma2_init = -2, 1
pi1_init, pi2_init = 0.5, 0.5  # 混合系数
def gaussian_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# EM算法
max_iter = 100
tolerance = 1e-6

for iteration in range(max_iter):
    # E步：计算每个数据点属于每个高斯分布的概率
    gamma1 = pi1_init * gaussian_pdf(data, mu1_init, sigma1_init)
    gamma2 = pi2_init * gaussian_pdf(data, mu2_init, sigma2_init)
    total = gamma1 + gamma2
    gamma1 /= total
    gamma2 /= total

    # M步：更新参数
    N1 = np.sum(gamma1)
    N2 = np.sum(gamma2)

    mu1_new = np.sum(gamma1 * data) / N1
    mu2_new = np.sum(gamma2 * data) / N2

    sigma1_new = np.sqrt(np.sum(gamma1 * (data - mu1_new) ** 2) / N1)
    sigma2_new = np.sqrt(np.sum(gamma2 * (data - mu2_new) ** 2) / N2)

    pi1_new = N1 / n_samples
    pi2_new = N2 / n_samples

    # 检查收敛
    if np.abs(mu1_new - mu1_init) < tolerance and np.abs(mu2_new - mu2_init) < tolerance:
        break

    # 更新参数
    mu1_init, mu2_init = mu1_new, mu2_new
    sigma1_init, sigma2_init = sigma1_new, sigma2_new
    pi1_init, pi2_init = pi1_new, pi2_new

print(f"Iteration {iteration}:")
print(f"mu1: {mu1_init}, sigma1: {sigma1_init}, pi1: {pi1_init}")
print(f"mu2: {mu2_init}, sigma2: {sigma2_init}, pi2: {pi2_init}")
# 可视化拟合结果
x = np.linspace(-5, 5, 1000)
plt.hist(data, bins=50, density=True, alpha=0.6, label='Data')
plt.plot(x, pi1_init * gaussian_pdf(x, mu1_init, sigma1_init), label='Gaussian 1')
plt.plot(x, pi2_init * gaussian_pdf(x, mu2_init, sigma2_init), label='Gaussian 2')
plt.title("Fitted Gaussian Mixture Model")
plt.legend()
plt.show()