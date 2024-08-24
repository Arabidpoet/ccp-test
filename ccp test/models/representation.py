import numpy as np
from integration import compute_projection_matrices, compute_svd
from covariance import compute_inverse_sqrt, compute_tilde_K12, compute_f_intraset_covariance, compute_f_interset_covariance

def transform_data(x1, x2, W1, W2):
    """
    将原始数据通过投影矩阵转换为潜在表示，并将两个视图的表示组合。

    参数:
    x1: 视图1的特征向量
    x2: 视图2的特征向量
    W1: 视图1的投影矩阵
    W2: 视图2的投影矩阵

    返回:
    y: 组合后的潜在表示
    """
    # 计算每个视图的潜在表示
    y1 = np.dot(W1.T, x1)
    y2 = np.dot(W2.T, x2)

    # 组合潜在表示
    y = np.concatenate((y1, y2))

    return y


# 示例
x1 = np.array([1, 2, 3])  # 视图1的特征
x2 = np.array([4, 5, 6])  # 视图2的特征

# 定义特征矩阵和参数
features1 = np.array([[1, 2], [3, 4], [5, 6]])
features2 = np.array([[7, 8], [9, 10], [11, 12]])
sigma = 1.0

# 计算协方差矩阵
K11_F = compute_f_intraset_covariance(features1, sigma)
K22_F = compute_f_intraset_covariance(features2, sigma)
K12_F = compute_f_interset_covariance(features1, features2, sigma)

# 计算 K_tilde_12
K_tilde_12 = compute_tilde_K12(K11_F, K12_F, K22_F)

# 计算 SVD
U, Sigma, Vt = compute_svd(K_tilde_12)

# V 是 Vt 的转置
V = Vt.T

# 设置降维维度 d
d = min(U.shape[1], V.shape[1])

#  W1 和 W2 是之前求解得到的投影矩阵
W1, W2 = compute_projection_matrices(K11_F, K22_F, U, V, d)

# 转换数据
y = transform_data(x1, x2, W1, W2)
print("组合后的潜在表示 y:", y) 