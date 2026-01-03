"""
pip install pandas numpy scikit-learn lifelines

cd C:/Users/Administrator/Desktop/CRLM
python model_performance_summary.py --input_dir results-SRC --group_path group.csv --output_path SRC-table/summary.csv

"""
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    recall_score, precision_score, f1_score
)
from sklearn.utils import resample
from lifelines.utils import concordance_index


def bootstrap_metric(y_true, y_score, metric_func, n_boot=1000, ci=95):
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = resample(range(n))
        boot_y_true = np.array(y_true)[idx]
        boot_y_score = np.array(y_score)[idx]
        try:
            score = metric_func(boot_y_true, boot_y_score)
            scores.append(score)
        except:
            continue
    if not scores:
        return np.nan, np.nan
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    return lower, upper

def calculate_metrics(y_true, y_score, threshold=0.5):
    y_true = np.array(y_true).flatten()
    y_score = np.array(y_score).flatten()
    
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[valid_idx]
    y_score = y_score[valid_idx]
    
    if len(y_true) < 2:
        return {k: np.nan for k in [
            "AUC", "AUC_95CI_lower", "AUC_95CI_upper",
            "Accuracy_mean", "Accuracy_std",
            "Sensitivity_mean", "Sensitivity_std",
            "Specificity_mean", "Specificity_std",
            "PPV_mean", "PPV_std", "NPV_mean", "NPV_std",
            "F1_mean", "F1_std", "C_index_mean", "C_index_std"
        ]}
    
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan
    auc_lower, auc_upper = bootstrap_metric(y_true, y_score, roc_auc_score) if len(set(y_true)) > 1 else (np.nan, np.nan)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = precision_score(y_true, y_pred) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1 = f1_score(y_true, y_pred) if (tp + fp + fn) > 0 else np.nan
    c_index = concordance_index(y_true, y_score) if len(set(y_true)) > 1 else np.nan
    
    n_boot = 1000
    boot_accuracy = []
    boot_sensitivity = []
    boot_specificity = []
    boot_ppv = []
    boot_npv = []
    boot_f1 = []
    boot_cindex = []
    
    for _ in range(n_boot):
        idx = resample(range(len(y_true)))
        boot_y_true = y_true[idx]
        boot_y_score = y_score[idx]
        boot_y_pred = (boot_y_score >= threshold).astype(int)
        
        try:
            boot_accuracy.append(accuracy_score(boot_y_true, boot_y_pred))
            boot_sensitivity.append(recall_score(boot_y_true, boot_y_pred) if (sum(boot_y_true) > 0) else np.nan)
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(boot_y_true, boot_y_pred).ravel()
            boot_specificity.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else np.nan)
            boot_ppv.append(precision_score(boot_y_true, boot_y_pred) if (sum(boot_y_pred) > 0) else np.nan)
            boot_npv.append(tn_b / (tn_b + fn_b) if (tn_b + fn_b) > 0 else np.nan)
            boot_f1.append(f1_score(boot_y_true, boot_y_pred) if (sum(boot_y_pred) + sum(boot_y_true) > 0) else np.nan)
            boot_cindex.append(concordance_index(boot_y_true, boot_y_score) if len(set(boot_y_true)) > 1 else np.nan)
        except:
            continue
    
    def mean_std(scores):
        scores = [s for s in scores if not np.isnan(s)]
        return np.mean(scores) if scores else np.nan, np.std(scores) if scores else np.nan
    
    acc_mean, acc_std = mean_std(boot_accuracy)
    sen_mean, sen_std = mean_std(boot_sensitivity)
    spe_mean, spe_std = mean_std(boot_specificity)
    ppv_mean, ppv_std = mean_std(boot_ppv)
    npv_mean, npv_std = mean_std(boot_npv)
    f1_mean, f1_std = mean_std(boot_f1)
    cindex_mean, cindex_std = mean_std(boot_cindex)
    
    return {
        "AUC": auc,
        "AUC_95CI_lower": auc_lower,
        "AUC_95CI_upper": auc_upper,
        "Accuracy_mean": acc_mean,
        "Accuracy_std": acc_std,
        "Sensitivity_mean": sen_mean,
        "Sensitivity_std": sen_std,
        "Specificity_mean": spe_mean,
        "Specificity_std": spe_std,
        "PPV_mean": ppv_mean,
        "PPV_std": ppv_std,
        "NPV_mean": npv_mean,
        "NPV_std": npv_std,
        "F1_mean": f1_mean,
        "F1_std": f1_std,
        "C_index_mean": cindex_mean,
        "C_index_std": cindex_std
    }

def parse_filename(filename):
    pattern = r'^(.+?)_(.+?)_(.+?)\.csv$'
    import re
    match = re.match(pattern, filename)
    return match.groups() if match else (None, None, None)

def load_true_labels(group_path):
    try:
        group_df = pd.read_csv(group_path)
        if 'ID' not in group_df.columns or 'label' not in group_df.columns:
            raise ValueError("group.csv must contain 'ID' and 'label' columns")
        return group_df.set_index('ID')['label'].to_dict()
    except Exception as e:
        print(f"[Error] 加载真实标签失败: {e}")
        raise


def process_performance(input_dir, group_path, output_path):

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    true_labels = load_true_labels(group_path)
    print(f"[Info] 成功加载 {len(true_labels)} 个样本的真实标签")
    
    results = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv') and filename != os.path.basename(group_path):
            model_type, method, dataset = parse_filename(filename)
            if not all([model_type, method, dataset]):
                print(f"[Warning] 文件名格式错误，跳过: {filename}")
                continue
            
            try:
                pred_df = pd.read_csv(os.path.join(input_dir, filename))
                if 'ID' not in pred_df.columns or 'label-1' not in pred_df.columns:
                    print(f"[Warning] 缺少必要列（ID/label-1），跳过: {filename}")
                    continue
                
                y_true = []
                y_score = []
                for _, row in pred_df.iterrows():
                    pid = row['ID']
                    if pid in true_labels:
                        true_label = true_labels[pid]
                        if isinstance(true_label, (int, float)) and not np.isnan(true_label):
                            y_true.append(true_label)
                        pred_score = row['label-1']
                        if isinstance(pred_score, (int, float)) and not np.isnan(pred_score):
                            y_score.append(pred_score)
                
                if len(y_true) < 2 or len(set(y_true)) < 1:
                    print(f"[Warning] 有效样本不足，跳过: {filename}")
                    continue
                
                metrics = calculate_metrics(y_true, y_score)
                
                result_row = {
                    "模型类别": model_type,
                    "建模方法": method,
                    "数据集": dataset,
                    **metrics
                }
                results.append(result_row)
                print(f"[Info] 已处理: {model_type}_{method}_{dataset} (有效样本数: {len(y_true)})")
            
            except Exception as e:
                print(f"[Error] 处理文件 {filename} 失败: {str(e)}")
                continue
    
    if results:
        result_df = pd.DataFrame(results)
        col_order = [
            "模型类别", "建模方法", "数据集",
            "AUC", "AUC_95CI_lower", "AUC_95CI_upper",
            "Accuracy_mean", "Accuracy_std",
            "Sensitivity_mean", "Sensitivity_std",
            "Specificity_mean", "Specificity_std",
            "PPV_mean", "PPV_std",
            "NPV_mean", "NPV_std",
            "F1_mean", "F1_std",
            "C_index_mean", "C_index_std"
        ]
        result_df = result_df[col_order]


        numeric_columns = result_df.select_dtypes(include=['float64', 'int64']).columns
        result_df[numeric_columns] = result_df[numeric_columns].round(3)
                
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[Success] 结果已保存到: {output_path}")
    else:
        print("[Error] 未处理到有效数据，无结果输出")


def main():
    parser = argparse.ArgumentParser(description='批量计算模型性能指标并输出汇总表格')
    parser.add_argument('--input_dir', required=True, help='模型预测结果CSV文件所在目录')
    parser.add_argument('--group_path', required=True, help='真实标签文件group.csv的路径')
    parser.add_argument('--output_path', required=True, help='输出汇总表格summary.csv的保存路径')
    args = parser.parse_args()
    
    process_performance(args.input_dir, args.group_path, args.output_path)

if __name__ == "__main__":
    main()