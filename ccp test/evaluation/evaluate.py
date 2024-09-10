from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_classification(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    auc = roc_auc_score(true_labels, predictions)
    return accuracy, f1, auc

# 输入: 预测标签, 真实标签
# 输出: 准确率, F1分数, AUC
def run_classification(X, y):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 随机森林分类器
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # 支持向量机分类器
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    # 评估模型
    rf_accuracy, rf_f1, rf_auc = evaluate_classification(rf_predictions, y_test)
    svm_accuracy, svm_f1, svm_auc = evaluate_classification(svm_predictions, y_test)

    # 输出评估结果
    print(f"Random Forest - Accuracy: {rf_accuracy:.2f}, F1 Score: {rf_f1:.2f}, AUC: {rf_auc:.2f}")
    print(f"SVM - Accuracy: {svm_accuracy:.2f}, F1 Score: {svm_f1:.2f}, AUC: {svm_auc:.2f}")


def run_clustering(X, n_clusters=2):
    # 使用 K 均值进行聚类
    #n_clusters = 2  # 设定聚类数
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # 使用 DBSCAN 进行聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)

    # 评估 K 均值聚类效果
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_labels)

    # 评估 DBSCAN 聚类效果
    dbscan_silhouette = silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
    dbscan_davies_bouldin = davies_bouldin_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

    # 输出聚类评估结果
    print(f"K-Means - Silhouette Score: {kmeans_silhouette:.2f}, Davies-Bouldin Score: {kmeans_davies_bouldin:.2f}")
    print(f"DBSCAN - Silhouette Score: {dbscan_silhouette:.2f}, Davies-Bouldin Score: {dbscan_davies_bouldin:.2f}")