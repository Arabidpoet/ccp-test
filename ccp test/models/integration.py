import numpy as np
from covariance import compute_inverse_sqrt, compute_tilde_K12, compute_f_intraset_covariance, compute_f_interset_covariance

def compute_svd(K_tilde_12):
    """
    计算 K_tilde_12 的奇异值分解
    K_tilde_12: 标准化后的协方差矩阵
    返回 U, Sigma, Vt (Vt 是 V 的转置)
    """
    U, Sigma, Vt = np.linalg.svd(K_tilde_12, full_matrices=False)
    return U, Sigma, Vt

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

# 计算投影矩阵 W1 和 W2
def compute_projection_matrices(K11_F, K22_F, U, V, d):
    """
    计算投影矩阵 W1 和 W2
    K11_F: 组内协方差矩阵 KF_11
    K22_F: 组内协方差矩阵 KF_22
    U: 奇异值分解中得到的 U 矩阵
    V: 奇异值分解中得到的 V 矩阵
    d: 降维维度

    返回:
    W1: 投影矩阵 W1
    W2: 投影矩阵 W2
    """
    # 计算 K11_F 和 K22_F 的逆平方根
    K11_inv_sqrt = compute_inverse_sqrt(K11_F)
    K22_inv_sqrt = compute_inverse_sqrt(K22_F)

    # 计算投影矩阵 W1 和 W2
    W1 = np.dot(K11_inv_sqrt, U[:, :d])
    W2 = np.dot(K22_inv_sqrt, V[:, :d])

    return W1, W2

# 计算投影矩阵 W1 和 W2
W1, W2 = compute_projection_matrices(K11_F, K22_F, U, V, d)

print("W1:", W1)
print("W2:", W2)