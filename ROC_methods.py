"""
cd C:/Users/Administrator/Desktop/CRLM


python ROC_methods.py --intotal_path SRC-table/intotal-cbind.csv --config_path SRC-ROC/roc.json --output_path SRC-ROC/roc_train.pdf --dataset Training



"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import argparse
import json
import chardet
from typing import Dict, List

# 颜色配置类
class Color:
    RESET = '\033[0m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'

def detect_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    result = chardet.detect(raw_data)
    return result['encoding'] or 'utf-8'

def load_config(config_path: str) -> Dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        required_keys = ['selected_columns', 'colors', 'positive_label']
        if not all(k in config for k in required_keys):
            raise ValueError(f"配置文件必须包含：{required_keys}")
        if len(config['selected_columns']) != len(config['colors']):
            raise ValueError("selected_columns和colors的长度必须一致")
        
        if 'show_ci' not in config:
            config['show_ci'] = True
            print(f"{Color.YELLOW}[Warning] 配置文件中未找到 'show_ci'，将默认显示95%CI。{Color.RESET}")
            
        return config
    except Exception as e:
        print(f"{Color.RED}[Error] 加载配置文件失败: {e}{Color.RESET}")
        raise

def bootstrap_auc(y_true, y_score, pos_label, n_boot=1000, ci=95) -> tuple:
    auc_scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = resample(range(n))
        y_true_sample = np.array(y_true)[idx]
        if len(np.unique(y_true_sample)) < 2:
            continue
        fpr, tpr, _ = roc_curve(
            y_true_sample, 
            np.array(y_score)[idx],
            pos_label=pos_label
        )
        auc_scores.append(auc(fpr, tpr))
    if not auc_scores:
        return np.nan, np.nan
        
    lower = np.percentile(auc_scores, (100 - ci) / 2)
    upper = np.percentile(auc_scores, 100 - (100 - ci) / 2)
    return round(lower, 3), round(upper, 3)

def plot_roc_from_horizontal(
    intotal_path: str,
    config: Dict,
    output_path: str,
    dataset_filter: str = None
):
    try:
        encoding = detect_encoding(intotal_path)
        print(f"{Color.BLUE}[Info] 检测到文件编码: {encoding}{Color.RESET}")
        df = pd.read_csv(intotal_path, encoding=encoding)
        required_cols = ['set', 'ID', 'true_label']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"横向表格必须包含列：{required_cols}")
    except Exception as e:
        print(f"{Color.RED}[Error] 读取intotal.csv失败: {e}{Color.RESET}")
        raise

    selected_cols = config['selected_columns']
    if selected_cols:
        first_column = selected_cols[0]
        title_prefix = first_column.split('_')[0]
    else:
        title_prefix = "Model"

    cohort_name = dataset_filter if dataset_filter else "All Cohorts"

    if dataset_filter:
        df = df[df['set'] == dataset_filter].copy()
        if len(df) == 0:
            raise ValueError(f"未找到数据集为 {dataset_filter} 的样本")
        print(f"{Color.BLUE}[Info] 已筛选数据集: {dataset_filter}（{len(df)}个样本）{Color.RESET}")
    else:
        print(f"{Color.BLUE}[Info] 使用所有数据集样本（{len(df)}个）{Color.RESET}")

    colors = config['colors']
    pos_label = config['positive_label']
    show_ci = config['show_ci']
    
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"横向表格中缺少配置的列：{missing_cols}")

    plt.figure(figsize=(8, 6))
    legend_items = []

    for col, color in zip(selected_cols, colors):
        valid_data = df[['true_label', col]].dropna()
        y_true = valid_data['true_label'].values
        y_score = valid_data[col].values

        if len(np.unique(y_true)) < 2:
            print(f"{Color.YELLOW}[Warning] 列 {col} 的标签类别不足，已跳过{Color.RESET}")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        roc_auc = round(auc(fpr, tpr), 3)
        

        parts = col.split('_')
        if len(parts) >= 2:
            display_name = parts[1] 
        else:

            display_name = col
        
        if show_ci:
            lower_ci, upper_ci = bootstrap_auc(y_true, y_score, pos_label)
            if np.isnan(lower_ci) or np.isnan(upper_ci):
                print(f"{Color.YELLOW}[Warning] 无法为 {col} 计算95% CI，可能是样本量不足。{Color.RESET}")
                legend_text = f"{display_name} AUC: {roc_auc}"
            else:
                legend_text = f"{display_name} AUC: {roc_auc} (95%CI: {lower_ci}-{upper_ci})"
        else:
            legend_text = f"{display_name} AUC: {roc_auc}"

        plt.plot(fpr, tpr, color=color, lw=2, label=legend_text)

    plt.plot([0, 1], [0, 1], color='#d3d3d3', lw=1.5, linestyle='--')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title(f"{title_prefix} Signature in {cohort_name} Cohort", fontsize=14)
    
    # 直接使用plt.legend()，因为plot时已经指定了label
    plt.legend(loc='lower right', fontsize=10)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path, format='pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"{Color.GREEN}[Success] ROC曲线已保存至: {output_path}{Color.RESET}")

def main():
    parser = argparse.ArgumentParser(description='从横向表格绘制自定义ROC曲线')
    parser.add_argument('--intotal_path', required=True, help='横向合并表intotal.csv的路径')
    parser.add_argument('--config_path', required=True, help='JSON配置文件路径')
    parser.add_argument('--output_path', required=True, help='输出ROC曲线PDF的路径')
    parser.add_argument('--dataset', help='可选：仅使用特定数据集的样本')
    args = parser.parse_args()

    try:
        import chardet
    except ImportError:
        print(f"{Color.YELLOW}[Warning] 缺少chardet库，正在自动安装...{Color.RESET}")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])

    config = load_config(args.config_path)
    print(f"{Color.BLUE}[Info] 已加载配置：{len(config['selected_columns'])}个预测列，正类标签为{config['positive_label']}，显示95%CI: {config['show_ci']}{Color.RESET}")

    plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
    plt.rcParams['axes.unicode_minus'] = False

    plot_roc_from_horizontal(
        intotal_path=args.intotal_path,
        config=config,
        output_path=args.output_path,
        dataset_filter=args.dataset
    )

if __name__ == "__main__":
    main()