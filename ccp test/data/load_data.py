import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

# 输入: 文件路径
# 输出: 标准化后的数据矩阵