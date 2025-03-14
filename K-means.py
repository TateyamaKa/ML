import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例数据
np.random.seed(0)
X = np.vstack([
    np.random.normal(loc=[5, 5], scale=1, size=(100, 2)),  # 第一个簇
    np.random.normal(loc=[15, 15], scale=1, size=(100, 2)),  # 第二个簇
    np.random.normal(loc=[25, 5], scale=1, size=(100, 2))   # 第三个簇
])

# K-means 聚类算法实现
def kmeans(X, k, max_iters=100):
    # 随机初始化簇的中心
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for i in range(max_iters):
        # 计算每个点到各个簇中心的距离
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        
        # 分配每个点到最近的簇
        labels = np.argmin(distances, axis=1)
        
        # 计算新的簇中心
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # 如果簇中心不再变化，则停止迭代
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

# 设置聚类数目
k = 3

# 运行 K-means 算法
centroids, labels = kmeans(X, k)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X')  # 聚类中心用红色标记
plt.title('K-means Clustering (without sklearn)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

