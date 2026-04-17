"""

分层bootstrap，1000次
有放回


自己改配置参数部分

"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix
)

# ================== 配置参数 ==================
# CSV文件路径
csv_path = "external_output.csv"  #

# CSV中的列名（根据实际列名修改）
col_patient_id = "PatientID"      # 患者ID列名
col_true_label = "True_Label"     # 真实标签列名（1=阳性，0=阴性）
col_pred_prob = "Pred_Probability" # 预测概率列名

# 分类阈值
threshold = 0.5 #自己调整，跟正文保持一致

# bootstrap次数
n_iterations = 1000
# =============================================









# ================== 读取数据 ==================
df = pd.read_csv(csv_path)

# 提取标签和预测概率
true_labels = df[col_true_label].values
pred_probs = df[col_pred_prob].values

# 验证数据完整性
print(f"成功读取 {len(df)} 条数据")
print(f"阳性样本数: {np.sum(true_labels == 1)}")
print(f"阴性样本数: {np.sum(true_labels == 0)}")
print(f"原始AUC: {roc_auc_score(true_labels, pred_probs):.4f}\n")
# =============================================

# 分离正负样本索引
pos_indices = np.where(true_labels == 1)[0]
neg_indices = np.where(true_labels == 0)[0]

n_pos = len(pos_indices)
n_neg = len(neg_indices)

# 存储结果
auc_list = []
prauc_list = []
sens_list = []
spec_list = []
prec_list = []
f1_list = []

print("正在进行 Bootstrap 重采样 (1000次)...")

for i in range(n_iterations):
    # ====== 分层bootstrap ======
    resampled_pos = np.random.choice(pos_indices, size=n_pos, replace=True)
    resampled_neg = np.random.choice(neg_indices, size=n_neg, replace=True)

    indices = np.concatenate([resampled_pos, resampled_neg])

    sample_labels = true_labels[indices]
    sample_probs = pred_probs[indices]

    # ====== AUC ======
    auc = roc_auc_score(sample_labels, sample_probs)
    auc_list.append(auc)

    # ====== PR-AUC ======
    prauc = average_precision_score(sample_labels, sample_probs)
    prauc_list.append(prauc)

    # ====== 分类结果 ======
    pred_labels = (sample_probs >= threshold).astype(int)

    # ====== 混淆矩阵 ======
    tn, fp, fn, tp = confusion_matrix(sample_labels, pred_labels).ravel()

    # ====== 指标 ======
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(sample_labels, pred_labels, zero_division=0)
    f1 = f1_score(sample_labels, pred_labels, zero_division=0)

    sens_list.append(sensitivity)
    spec_list.append(specificity)
    prec_list.append(precision)
    f1_list.append(f1)


# ====== 统计函数 ======
def summarize(metric_list):
    arr = np.array(metric_list)
    mean = np.mean(arr)
    ci_lower = np.percentile(arr, 2.5)
    ci_upper = np.percentile(arr, 97.5)
    std_error = np.std(arr)
    return mean, ci_lower, ci_upper, std_error


# ====== 输出结果 ======
metrics = {
    "AUC": auc_list,
    "PR-AUC": prauc_list,
    "Sensitivity": sens_list,
    "Specificity": spec_list,
    "Precision": prec_list,
    "F1-score": f1_list
}

print("\n===== Bootstrap Results (1000 iterations) =====\n")

for name, values in metrics.items():
    mean, ci_l, ci_u, se = summarize(values)
    print(f"{name}:")
    print(f"  Mean = {mean:.4f}")
    print(f"  95% CI = ({ci_l:.4f}, {ci_u:.4f})")
    print(f"  Std Error = {se:.4f}\n")

# ====== 可选：保存结果到CSV ======
results_df = pd.DataFrame({
    "Metric": list(metrics.keys()),
    "Mean": [summarize(v)[0] for v in metrics.values()],
    "CI_Lower": [summarize(v)[1] for v in metrics.values()],
    "CI_Upper": [summarize(v)[2] for v in metrics.values()],
    "Std_Error": [summarize(v)[3] for v in metrics.values()]
})
results_df.to_csv("bootstrap_results.csv", index=False)
print("结果已保存至 bootstrap_results.csv")