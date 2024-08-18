def transform_data(data, projection_matrix):
    return np.dot(data, projection_matrix)

# 输入: 原始数据, 投影矩阵
# 输出: 转换后的表示