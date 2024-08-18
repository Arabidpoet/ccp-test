from data.load_data import load_and_preprocess_data
from models.nonlinear_mapping import nonlinear_mapping
from models.covariance import compute_f_intraset_covariance, compute_f_interset_covariance
from models.projection import build_projection_matrix
from models.representation import transform_data
from evaluation.evaluate import evaluate_classification

# 加载数据
view1 = load_and_preprocess_data('data/view1.csv')
view2 = load_and_preprocess_data('data/view2.csv')

# 非线性映射
mapped_view1 = nonlinear_mapping(view1)
mapped_view2 = nonlinear_mapping(view2)

# 计算协方差矩阵
f_intraset_cov = compute_f_intraset_covariance(mapped_view1)
f_interset_cov = compute_f_interset_covariance(mapped_view1, mapped_view2)

# 构建投影矩阵
projection_matrix = build_projection_matrix(f_intraset_cov + f_interset_cov)

# 表示学习
latent_representation1 = transform_data(mapped_view1, projection_matrix)

# 评估
true_labels = [0, 1, 0, 1]  # 示例标签
predictions = [0, 1, 0, 1]  # 示例预测
accuracy, f1, auc = evaluate_classification(predictions, true_labels)
print(f"Accuracy: {accuracy}, F1 Score: {f1}, AUC: {auc}")

