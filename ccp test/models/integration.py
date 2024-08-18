from scipy.linalg import eigh

def build_projection_matrix(cov_matrix):
    # 选择前k个最大特征值对应的特征向量
    eigvals, eigvecs = eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    return eigvecs[:, idx]

# 输入: 协变矩阵
# 输出: 投影矩阵