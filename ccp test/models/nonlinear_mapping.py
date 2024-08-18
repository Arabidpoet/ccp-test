from sklearn.decomposition import KernelPCA

def nonlinear_mapping(data, kernel='rbf', n_components=10):
    kpca = KernelPCA(kernel=kernel, n_components=n_components)
    return kpca.fit_transform(data)

# 输入: 数据矩阵, 核类型, 成分数
# 输出: 非线性映射后的特征