#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# 基本运行
python DCA-guided_threshold.py --config ./DCA-guided_threshold/DCA-guided_threshold.yaml

# 自定义参数
python DCA-guided_threshold.py \
  --config ./DCA-guided_threshold/DCA-guided_threshold.yaml \
  --pt_start 0.05 --pt_end 0.30 \
  --iter 500 \
  --seed 123


python DCA-guided_threshold.py  --config ./DCA-guided_threshold/DCA-guided_threshold.yaml   --pt_start 0.05 --pt_end 0.30  --iter 500  --seed 123



"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, precision_score, f1_score
import os
import sys

try:
    import yaml
except ImportError:
    print("请先安装PyYAML: pip install pyyaml")
    sys.exit(1)


def parse_arguments():
    parser = argparse.ArgumentParser(description='DCA-based threshold optimization and validation')
    parser.add_argument('--config', type=str, default='dca_threshold_analysis.yaml', help='Config file path')
    parser.add_argument('--pt_start', type=float, default=None, help='Treatment threshold start (override config)')
    parser.add_argument('--pt_end', type=float, default=None, help='Treatment threshold end (override config)')
    parser.add_argument('--iter', type=int, default=None, help='Bootstrap iterations (override config)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (override config)')
    return parser.parse_args()


def load_config(config_path):
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def calculate_net_benefit(tp, fp, n, pt):
    """Calculate net benefit at a given treatment threshold pt"""
    if pt >= 1.0:
        return 0
    return (tp / n) - (fp / n) * (pt / (1 - pt))


def find_optimal_thresholds(train_labels, train_probs, pt_range, threshold_range):
    """
    Find optimal model prediction thresholds for each treatment threshold (pt)
    Based on maximizing net benefit on training cohort
    """
    n = len(train_labels)
    
    results = []
    best_net_benefit = -np.inf
    best_pt = None
    best_threshold = None
    
    for pt in pt_range:
        best_nb_for_pt = -np.inf
        best_t_for_pt = None
        
        for t in threshold_range:
            pred_labels = (train_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(train_labels, pred_labels).ravel()
            
            nb = calculate_net_benefit(tp, fp, n, pt)
            
            if nb > best_nb_for_pt:
                best_nb_for_pt = nb
                best_t_for_pt = t
        
        results.append({
            'pt': pt,
            'optimal_threshold': best_t_for_pt,
            'max_net_benefit': best_nb_for_pt
        })
        
        if best_nb_for_pt > best_net_benefit:
            best_net_benefit = best_nb_for_pt
            best_pt = pt
            best_threshold = best_t_for_pt
    
    return {
        'optimal_pt': best_pt,
        'optimal_threshold': best_threshold,
        'max_net_benefit': best_net_benefit,
        'all_results': results
    }


def find_youden_threshold(train_labels, train_probs, threshold_range):
    """Find threshold that maximizes Youden index (sensitivity + specificity - 1)"""
    best_youden = -1
    best_threshold = 0.5
    best_sensitivity = 0
    best_specificity = 0
    
    for t in threshold_range:
        pred_labels = (train_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(train_labels, pred_labels).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sensitivity + specificity - 1
        
        if youden > best_youden:
            best_youden = youden
            best_threshold = t
            best_sensitivity = sensitivity
            best_specificity = specificity
    
    return {
        'threshold': best_threshold,
        'youden_index': best_youden,
        'sensitivity': best_sensitivity,
        'specificity': best_specificity
    }


def find_sensitivity_threshold(train_labels, train_probs, threshold_range, target_sensitivity=0.95):
    """Find threshold that achieves at least target sensitivity (lowest possible)"""
    best_threshold = 0.5
    best_sensitivity = 0
    
    for t in threshold_range:
        pred_labels = (train_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(train_labels, pred_labels).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if sensitivity >= target_sensitivity and sensitivity > best_sensitivity:
            best_sensitivity = sensitivity
            best_threshold = t
    
    return {
        'threshold': best_threshold,
        'sensitivity': best_sensitivity
    }


def find_specificity_threshold(train_labels, train_probs, threshold_range, target_specificity=0.85):
    """Find threshold that achieves at least target specificity (highest possible)"""
    best_threshold = 0.5
    best_specificity = 0
    
    for t in threshold_range:
        pred_labels = (train_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(train_labels, pred_labels).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if specificity >= target_specificity and specificity > best_specificity:
            best_specificity = specificity
            best_threshold = t
    
    return {
        'threshold': best_threshold,
        'specificity': best_specificity
    }


def bootstrap_metrics(true_labels, pred_probs, threshold, n_iterations=1000, random_seed=42):
    """Calculate bootstrap confidence intervals for multiple metrics"""
    np.random.seed(random_seed)
    
    pos_indices = np.where(true_labels == 1)[0]
    neg_indices = np.where(true_labels == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    if n_pos == 0 or n_neg == 0:
        return None
    
    auc_list = []
    prauc_list = []
    sens_list = []
    spec_list = []
    prec_list = []
    f1_list = []
    acc_list = []
    youden_list = []
    
    for _ in range(n_iterations):
        resampled_pos = np.random.choice(pos_indices, size=n_pos, replace=True)
        resampled_neg = np.random.choice(neg_indices, size=n_neg, replace=True)
        indices = np.concatenate([resampled_pos, resampled_neg])
        
        sample_labels = true_labels[indices]
        sample_probs = pred_probs[indices]
        
        try:
            auc_list.append(roc_auc_score(sample_labels, sample_probs))
            prauc_list.append(average_precision_score(sample_labels, sample_probs))
        except:
            continue
        
        pred_labels = (sample_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(sample_labels, pred_labels).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = precision_score(sample_labels, pred_labels, zero_division=0)
        f1 = f1_score(sample_labels, pred_labels, zero_division=0)
        acc = (tp + tn) / len(sample_labels)
        youden = sens + spec - 1
        
        sens_list.append(sens)
        spec_list.append(spec)
        prec_list.append(prec)
        f1_list.append(f1)
        acc_list.append(acc)
        youden_list.append(youden)
    
    def summarize(arr):
        return {
            'mean': np.mean(arr),
            'ci_lower': np.percentile(arr, 2.5),
            'ci_upper': np.percentile(arr, 97.5),
            'std_error': np.std(arr)
        }
    
    return {
        'AUC': summarize(auc_list),
        'PR-AUC': summarize(prauc_list),
        'Accuracy': summarize(acc_list),
        'Sensitivity': summarize(sens_list),
        'Specificity': summarize(spec_list),
        'Precision': summarize(prec_list),
        'F1-score': summarize(f1_list),
        'Youden': summarize(youden_list),
        'original': {
            'AUC': roc_auc_score(true_labels, pred_probs),
            'PR-AUC': average_precision_score(true_labels, pred_probs),
            'n_samples': len(true_labels),
            'n_positive': n_pos,
            'n_negative': n_neg
        }
    }


def evaluate_at_threshold(true_labels, pred_probs, threshold):
    """Evaluate model performance at a given threshold"""
    pred_labels = (pred_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = (tp + tn) / len(true_labels)
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    youden = sensitivity + specificity - 1
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'accuracy': accuracy,
        'f1': f1,
        'youden': youden,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def main():
    args = parse_arguments()
    config = load_config(args.config)
    
    # 获取配置参数
    excel_path = config['data']['excel_path']
    sheet_name = config['data'].get('sheet_name')
    col_group = config['columns']['group']
    col_true_label = config['columns']['true_label']
    col_pred_prob = config['columns']['pred_prob']
    
    # DCA参数
    pt_start = args.pt_start or config['dca']['pt_range']['start']
    pt_end = args.pt_end or config['dca']['pt_range']['end']
    pt_step = config['dca']['pt_range']['step']
    thresh_start = config['dca']['threshold_range']['start']
    thresh_end = config['dca']['threshold_range']['end']
    thresh_step = config['dca']['threshold_range']['step']
    
    # Bootstrap参数
    n_iterations = args.iter or config['bootstrap']['n_iterations']
    random_seed = args.seed or config['bootstrap']['random_seed']
    
    # 输出目录
    output_dir = config['output']['dir']
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("DCA-based Threshold Optimization and Validation")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Excel: {excel_path}")
    print(f"Pt range: {pt_start} - {pt_end} (step {pt_step})")
    print(f"Threshold range: {thresh_start} - {thresh_end} (step {thresh_step})")
    print(f"Bootstrap iterations: {n_iterations}")
    print(f"Random seed: {random_seed}")
    print("=" * 70)
    
    # 读取数据
    if sheet_name:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(excel_path)
    print(f"\nLoaded {len(df)} rows from Excel")
    
    # 检查列
    required_cols = [col_group, col_true_label, col_pred_prob]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found. Available: {list(df.columns)}")
            sys.exit(1)
    
    # 分离数据集
    train_df = df[df[col_group] == 'train']
    internal_df = df[df[col_group] == 'internal']
    external_df = df[df[col_group] == 'external']
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Internal validation: {len(internal_df)} samples")
    print(f"  External validation: {len(external_df)} samples")
    
    # 提取训练集数据
    train_labels = train_df[col_true_label].values
    train_probs = train_df[col_pred_prob].values
    
    # ================== 步骤1: DCA分析找最优阈值 ==================
    print("\n" + "=" * 70)
    print("Step 1: DCA Analysis on Training Cohort")
    print("=" * 70)
    
    pt_range = np.arange(pt_start, pt_end + pt_step, pt_step)
    threshold_range = np.arange(thresh_start, thresh_end + thresh_step, thresh_step)
    
    optimal = find_optimal_thresholds(train_labels, train_probs, pt_range, threshold_range)
    
    optimal_pt = optimal['optimal_pt']
    optimal_threshold = optimal['optimal_threshold']
    
    print(f"\nOptimal treatment threshold (pt): {optimal_pt:.3f}")
    print(f"Optimal model prediction threshold: {optimal_threshold:.3f}")
    print(f"Maximum net benefit: {optimal['max_net_benefit']:.4f}")
    
    # 保存DCA结果
    dca_df = pd.DataFrame(optimal['all_results'])
    dca_df.to_csv(os.path.join(output_dir, 'dca_results.csv'), index=False)
    
    # ================== 步骤2: Youden指数最优阈值 ==================
    print("\n" + "=" * 70)
    print("Step 2: Youden Index Analysis on Training Cohort")
    print("=" * 70)
    
    youden_result = find_youden_threshold(train_labels, train_probs, threshold_range)
    
    print(f"\nYouden-optimized threshold: {youden_result['threshold']:.3f}")
    print(f"Youden index: {youden_result['youden_index']:.4f}")
    print(f"  Sensitivity: {youden_result['sensitivity']:.3f} ({youden_result['sensitivity']*100:.1f}%)")
    print(f"  Specificity: {youden_result['specificity']:.3f} ({youden_result['specificity']*100:.1f}%)")
    
    # ================== 步骤3: 场景特定阈值 ==================
    print("\n" + "=" * 70)
    print("Step 3: Scenario-Specific Thresholds")
    print("=" * 70)
    
    # 高灵敏度阈值 (sensitivity >= 95%)
    sens_result = find_sensitivity_threshold(train_labels, train_probs, threshold_range, target_sensitivity=0.95)
    
    # 高特异性阈值 (specificity >= 85%)
    spec_result = find_specificity_threshold(train_labels, train_probs, threshold_range, target_specificity=0.85)
    
    print(f"\nHigh-sensitivity threshold (sensitivity >= 95%): {sens_result['threshold']:.4f}")
    print(f"  Achieved sensitivity: {sens_result['sensitivity']:.3f} ({sens_result['sensitivity']*100:.1f}%)")
    print(f"\nHigh-specificity threshold (specificity >= 85%): {spec_result['threshold']:.4f}")
    print(f"  Achieved specificity: {spec_result['specificity']:.3f} ({spec_result['specificity']*100:.1f}%)")
    
    # ================== 步骤4: 在验证集上评估 ==================
    print("\n" + "=" * 70)
    print("Step 4: Model Evaluation on Validation Cohorts")
    print("=" * 70)
    
    # 定义要评估的阈值列表
    thresholds_to_evaluate = [
        ('DCA-Optimized', optimal_threshold),
        ('Youden-Optimized', youden_result['threshold']),
        ('High Sensitivity (>=95%)', sens_result['threshold']),
        ('High Specificity (>=85%)', spec_result['threshold'])
    ]
    
    all_results = []
    
    for threshold_name, threshold_value in thresholds_to_evaluate:
        for val_name, val_df in [('Internal', internal_df), ('External', external_df)]:
            if len(val_df) == 0:
                continue
            
            labels = val_df[col_true_label].values
            probs = val_df[col_pred_prob].values
            
            # Bootstrap CI
            bootstrap_results = bootstrap_metrics(labels, probs, threshold_value, n_iterations, random_seed)
            
            # 点估计
            point_est = evaluate_at_threshold(labels, probs, threshold_value)
            
            if bootstrap_results:
                all_results.append({
                    'Dataset': val_name,
                    'Threshold_Type': threshold_name,
                    'Threshold_Value': threshold_value,
                    'N_Samples': len(val_df),
                    'N_Positive': np.sum(labels == 1),
                    'N_Negative': np.sum(labels == 0),
                    'AUC_Point': bootstrap_results['original']['AUC'],
                    'AUC_Mean': bootstrap_results['AUC']['mean'],
                    'AUC_CI_Lower': bootstrap_results['AUC']['ci_lower'],
                    'AUC_CI_Upper': bootstrap_results['AUC']['ci_upper'],
                    'Sensitivity_Point': point_est['sensitivity'],
                    'Sensitivity_Mean': bootstrap_results['Sensitivity']['mean'],
                    'Sensitivity_CI_Lower': bootstrap_results['Sensitivity']['ci_lower'],
                    'Sensitivity_CI_Upper': bootstrap_results['Sensitivity']['ci_upper'],
                    'Specificity_Point': point_est['specificity'],
                    'Specificity_Mean': bootstrap_results['Specificity']['mean'],
                    'Specificity_CI_Lower': bootstrap_results['Specificity']['ci_lower'],
                    'Specificity_CI_Upper': bootstrap_results['Specificity']['ci_upper'],
                    'Youden_Point': point_est['youden'],
                    'Youden_Mean': bootstrap_results['Youden']['mean'],
                    'Youden_CI_Lower': bootstrap_results['Youden']['ci_lower'],
                    'Youden_CI_Upper': bootstrap_results['Youden']['ci_upper'],
                    'Accuracy_Point': point_est['accuracy'],
                    'Precision_Point': point_est['precision'],
                    'F1_Point': point_est['f1']
                })
    
    # ================== 保存结果 ==================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    
    # 保存汇总结果
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(output_dir, 'validation_results.csv'), index=False)
    
    # 保存最优阈值信息
    threshold_info = pd.DataFrame([
        {'Scenario': 'DCA-Optimized (Maximum Net Benefit)', 
         'Threshold': optimal_threshold,
         'Optimal_Pt': optimal_pt,
         'Youden_Index': None},
        {'Scenario': 'Youden-Optimized (Balance Sens/Spec)',
         'Threshold': youden_result['threshold'],
         'Optimal_Pt': None,
         'Youden_Index': youden_result['youden_index']},
        {'Scenario': 'High Sensitivity (Minimize Missed Diagnosis)',
         'Threshold': sens_result['threshold'],
         'Optimal_Pt': None,
         'Youden_Index': None},
        {'Scenario': 'High Specificity (Reduce Overtreatment)',
         'Threshold': spec_result['threshold'],
         'Optimal_Pt': None,
         'Youden_Index': None}
    ])
    threshold_info.to_csv(os.path.join(output_dir, 'optimal_thresholds.csv'), index=False)
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - dca_results.csv: DCA analysis details")
    print(f"  - optimal_thresholds.csv: Four clinically meaningful thresholds")
    print(f"  - validation_results.csv: Performance on validation cohorts")
    
    # ================== 打印最终汇总 ==================
    print("\n" + "=" * 70)
    print("SUMMARY FOR RESPONSE TO REVIEWER")
    print("=" * 70)
    
    print(f"\n1. Thresholds Derived from Training Cohort (N=178):")
    print(f"   - DCA-optimized (max net benefit): {optimal_threshold:.3f} (pt={optimal_pt:.3f})")
    print(f"   - Youden-optimized (max sens+spec-1): {youden_result['threshold']:.3f} (Youden index={youden_result['youden_index']:.4f})")
    print(f"   - High sensitivity (>=95%): {sens_result['threshold']:.4f}")
    print(f"   - High specificity (>=85%): {spec_result['threshold']:.4f}")
    
    print(f"\n2. Performance on External Validation Cohort (N=109, ER+={np.sum(external_df[col_true_label]==1)}):")
    
    for threshold_name, threshold_value in thresholds_to_evaluate:
        # 找到对应的外部验证结果
        ext_result = next((r for r in all_results if r['Dataset'] == 'External' and r['Threshold_Type'] == threshold_name), None)
        if ext_result:
            print(f"\n   {threshold_name} (threshold={threshold_value:.3f}):")
            print(f"     AUC: {ext_result['AUC_Point']:.4f} (95% CI: {ext_result['AUC_CI_Lower']:.4f}-{ext_result['AUC_CI_Upper']:.4f})")
            print(f"     Sensitivity: {ext_result['Sensitivity_Point']:.3f} ({ext_result['Sensitivity_Point']*100:.1f}%)")
            print(f"     Specificity: {ext_result['Specificity_Point']:.3f} ({ext_result['Specificity_Point']*100:.1f}%)")
            if threshold_name == 'Youden-Optimized':
                print(f"     Youden index: {ext_result['Youden_Point']:.4f}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()