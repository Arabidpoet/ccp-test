from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import fractional_matrix_power
import numpy as np

def gaussian_kernel(X, Y, sigma):
    """计算高斯核矩阵"""
    return rbf_kernel(X, Y, gamma=1/(2*sigma**2))

def compute_f_intraset_covariance(features, sigma):
    """
    计算F-intraset协变矩阵。
    features: 输入的特征矩阵, 形状为 (n_samples, n_features)
    sigma: 高斯核的宽度参数
    返回: F-intraset协变矩阵, 形状为 (n_samples, n_samples)
    """
    KF = gaussian_kernel(features, features, sigma)
    return KF

def compute_f_interset_covariance(features1, features2, sigma):
    """
    计算F-interset协变矩阵。
    features1: 第一个视图的特征矩阵, 形状为 (n_samples_1, n_features)
    features2: 第二个视图的特征矩阵, 形状为 (n_samples_2, n_features)
    sigma: 高斯核的宽度参数
    返回: F-interset协变矩阵, 形状为 (n_samples_1, n_samples_2)
    """
    KF = gaussian_kernel(features1, features2, sigma)
    return KF

def compute_inverse_sqrt(matrix):
    """
    计算矩阵的逆平方根。
    matrix: 输入矩阵
    返回: 矩阵的逆平方根
    """
    return fractional_matrix_power(matrix, -0.5)

def compute_tilde_K12(K11, K12, K22):
    """
    计算 K̃12 矩阵。
    K11: 组内协变矩阵 K11^F
    K12: 组间协变矩阵 K12^F
    K22: 组内协变矩阵 K22^F
    返回: 计算得到的 K̃12 矩阵
    """
    K11_inv_sqrt = compute_inverse_sqrt(K11)
    K22_inv_sqrt = compute_inverse_sqrt(K22)
    K_tilde_12 = K11_inv_sqrt @ K12 @ K22_inv_sqrt
    return K_tilde_12


if __name__ == "__main__":
    # 示例数据
    features1 = np.array([[1, 2], [3, 4], [5, 6]])
    features2 = np.array([[7, 8], [9, 10], [11, 12]])
    sigma = 1.0

    # 计算F-intraset和F-interset协变矩阵
    K11_F = compute_f_intraset_covariance(features1, sigma)
    K22_F = compute_f_intraset_covariance(features2, sigma)
    K12_F = compute_f_interset_covariance(features1, features2, sigma)

    # 计算 K̃12 矩阵
    K_tilde_12 = compute_tilde_K12(K11_F, K12_F, K22_F)

    print("K11^F:\n", K11_F)
    print("K22^F:\n", K22_F)
    print("K12^F:\n", K12_F)
    print("K̃12:\n", K_tilde_12)