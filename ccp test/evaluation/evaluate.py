from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_classification(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    auc = roc_auc_score(true_labels, predictions)
    return accuracy, f1, auc

# 输入: 预测标签, 真实标签
# 输出: 准确率, F1分数, AUC