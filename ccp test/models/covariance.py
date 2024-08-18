import numpy as np

def compute_f_intraset_covariance(features):
    return np.cov(features, rowvar=False)

def compute_f_interset_covariance(features1, features2):
    return np.cov(features1.T, features2.T)

# 输入: 特征矩阵
# 输出: F-intraset 或 F-interset 协变矩阵