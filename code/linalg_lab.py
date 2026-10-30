import numpy as np

np.random.seed(0)

# 1. SVD
A = np.array([[3.0, 1.0],
              [1.0, 3.0]])
U, S, Vt = np.linalg.svd(A)
print("U =\n", U)
print("S =", S)
print("Vt =\n", Vt)

# 2. PCA
X = np.random.randn(100, 2) @ np.array([[2.0, 0.5],
                                        [0.5, 1.0]])
Xc = X - X.mean(axis=0)
U2, S2, Vt2 = np.linalg.svd(Xc, full_matrices=False)
print("PCA 第一主方向 =", Vt2[0])
print("前两个奇异值 =", S2[:2])

# 3. 最小二乘
x = np.linspace(0, 1, 20)
y = 2.0 * x + 1.0 + 0.1 * np.random.randn(20)
A2 = np.vstack([x, np.ones_like(x)]).T
coef, *_ = np.linalg.lstsq(A2, y, rcond=None)
print("拟合的 a, b =", coef)
