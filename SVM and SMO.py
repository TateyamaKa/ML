import random
import matplotlib.pyplot as plt

# 生成线性可分的数据
def generate_data(n_samples=40):
    X = []
    y = []
    # 使用分隔线 2x + 3y + 1 = 0 生成标签
    for _ in range(n_samples):
        x1 = random.uniform(-5, 5)
        x2 = random.uniform(-5, 5)
        val = 2 * x1 + 3 * x2 + 1
        if val > 0:
            y_i = 1
        else:
            y_i = -1
        X.append([x1, x2])
        y.append(y_i)
    return X, y

# SVM类实现SMO算法
class SVM:
    def __init__(self, C=1.0, tol=0.001, max_iter=100):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = []
        self.b = 0
        self.X = []
        self.y = []
        self.errors = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples = len(X)
        self.alpha = [0.0] * n_samples
        self.b = 0.0
        self.errors = [self.decision_function(X[i]) - y[i] for i in range(n_samples)]
        iter_ = 0
        examine_all = True
        num_changed = 0
        while iter_ < self.max_iter and (num_changed > 0 or examine_all):
            num_changed = 0
            if examine_all:
                for i in range(n_samples):
                    num_changed += self.examine_example(i)
            else:
                for i in range(n_samples):
                    if 0 < self.alpha[i] < self.C:
                        num_changed += self.examine_example(i)
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            iter_ += 1

    def examine_example(self, i):
        y_i = self.y[i]
        alpha_i = self.alpha[i]
        E_i = self.errors[i]
        r_i = E_i * y_i
        if (r_i < -self.tol and alpha_i < self.C) or (r_i > self.tol and alpha_i > 0):
            j = self.select_j(i, E_i)
            if j != -1:
                return self.update_alpha(i, j)
        return 0

    def select_j(self, i, E_i):
        max_delta = 0
        j = -1
        for idx in range(len(self.alpha)):
            if idx != i:
                E_j = self.decision_function(self.X[idx]) - self.y[idx]
                delta = abs(E_i - E_j)
                if delta > max_delta:
                    max_delta = delta
                    j = idx
        if j != -1:
            return j
        return random.choice([k for k in range(len(self.alpha)) if k != i])

    def update_alpha(self, i, j):
        X_i, X_j = self.X[i], self.X[j]
        y_i, y_j = self.y[i], self.y[j]
        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
        if y_i != y_j:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_j_old + alpha_i_old - self.C)
            H = min(self.C, alpha_j_old + alpha_i_old)
        if L == H:
            return 0
        eta = 2 * self.kernel(X_i, X_j) - self.kernel(X_i, X_i) - self.kernel(X_j, X_j)
        if eta >= 0:
            return 0
        E_i = self.decision_function(X_i) - y_i
        E_j = self.decision_function(X_j) - y_j
        alpha_j_new = alpha_j_old - y_j * (E_i - E_j) / eta
        if alpha_j_new > H:
            alpha_j_new = H
        elif alpha_j_new < L:
            alpha_j_new = L
        if abs(alpha_j_new - alpha_j_old) < 1e-5:
            return 0
        alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new)
        self.alpha[i], self.alpha[j] = alpha_i_new, alpha_j_new
        b_i = self.b - E_i - y_i * (alpha_i_new - alpha_i_old) * self.kernel(X_i, X_i) - y_j * (alpha_j_new - alpha_j_old) * self.kernel(X_i, X_j)
        b_j = self.b - E_j - y_i * (alpha_i_new - alpha_i_old) * self.kernel(X_i, X_j) - y_j * (alpha_j_new - alpha_j_old) * self.kernel(X_j, X_j)
        if 0 < alpha_i_new < self.C:
            self.b = b_i
        elif 0 < alpha_j_new < self.C:
            self.b = b_j
        else:
            self.b = (b_i + b_j) / 2
        self.errors[i] = self.decision_function(X_i) - y_i
        self.errors[j] = self.decision_function(X_j) - y_j
        return 1

    def kernel(self, x1, x2):
        return sum([a * b for a, b in zip(x1, x2)])

    def decision_function(self, x):
        result = 0.0
        for i in range(len(self.alpha)):
            result += self.alpha[i] * self.y[i] * self.kernel(self.X[i], x)
        return result + self.b

    def get_weights(self):
        if not self.X:
            return []
        weights = [0.0] * len(self.X[0])
        for i in range(len(self.alpha)):
            if self.alpha[i] > 0:
                for d in range(len(weights)):
                    weights[d] += self.alpha[i] * self.y[i] * self.X[i][d]
        return weights

# 生成数据
X, y = generate_data(n_samples=40)

# 训练SVM
svm = SVM(C=1.0, tol=0.001, max_iter=100)
svm.fit(X, y)

# 获取权重和截距
weights = svm.get_weights()
b = svm.b

# 绘制数据点
plt.figure(figsize=(10, 6))
for i, (x, label) in enumerate(zip(X, y)):
    color = 'blue' if label == 1 else 'red'
    marker = 'o' if label == 1 else 'x'
    plt.scatter(x[0], x[1], color=color, marker=marker, label=f'Class {label}' if i ==0 else "")

# 绘制决策边界
x_min, x_max = min(x[0] for x in X), max(x[0] for x in X)
y_min, y_max = min(x[1] for x in X), max(x[1] for x in X)
xx = [x_min -1, x_max +1]
if weights[1] != 0:
    yy = [(-b - weights[0] * x) / weights[1] for x in xx]
    plt.plot(xx, yy, 'k-', label='Decision Boundary')
else:
    plt.axvline(x=-b/weights[0], color='k', label='Decision Boundary')

# 标记支持向量
support_vectors = [X[i] for i in range(len(X)) if svm.alpha[i] > 0]
for sv in support_vectors:
    plt.scatter(sv[0], sv[1], s=100, facecolors='none', edgecolors='black', label='Support Vectors' if sv == support_vectors[0] else "")

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()