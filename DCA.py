"""



pip install pandas numpy matplotlib scikit-learn


python DCA.py --config SRC-DCA/DCA.json


"""


import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from matplotlib.colors import hex2color, rgb2hex

plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

def calculate_net_benefit(y_true, y_pred_proba, thresholds):

    n = len(y_true)
    net_benefits = np.zeros_like(thresholds, dtype=float)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    
    prevalence = np.mean(y_true)
    
    for i, thresh in enumerate(thresholds):
        idx = np.searchsorted(_[::-1], thresh)
        idx = len(_) - 1 - idx
        
        if idx < len(fpr):
            current_tpr = tpr[idx]
            current_fpr = fpr[idx]
        else:
            current_tpr = 0
            current_fpr = 0

        if thresh == 1:
            net_benefit = 0.0
        else:
            net_benefit = (current_tpr * prevalence) - (current_fpr * (1 - prevalence) * (thresh / (1 - thresh)))
        net_benefits[i] = net_benefit

    return net_benefits

def plot_dca_for_dataset(df, dataset_name, models, threshold_range, output_path, color):

    print(f"[INFO] 正在为数据集 '{dataset_name}' 绘制DCA曲线...")
    
    df_dataset = df[df['set'] == dataset_name].copy()
    if df_dataset.empty:
        print(f"[WARNING] 未找到数据集 '{dataset_name}' 的数据，已跳过。")
        return

    thresholds = np.arange(threshold_range[0], threshold_range[1], 0.01)
    
    plt.figure(figsize=(10, 8))
    
    plt.axhline(y=0, color='gray', linestyle='--', label='None')
    all_benefit = np.mean(df_dataset['true_label'])
    plt.axhline(y=all_benefit, color='gray', linestyle=':', label='All')
    
    for i, model in enumerate(models):
        y_true = df_dataset['true_label']
        y_pred_proba = df_dataset[model['prediction_col']]
        
        net_benefits = calculate_net_benefit(y_true, y_pred_proba, thresholds)
        
        plt.plot(thresholds, net_benefits, 
                 label=f"{model['name']} ({model['algorithm']})", 
                 color=color, linewidth=2)

    plt.title(f'Decision Curve Analysis - {dataset_name}', fontsize=16)
    plt.xlabel('Threshold Probability', fontsize=14)
    plt.ylabel('Net Benefit', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 曲线已成功保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='根据配置文件绘制决策曲线分析 (DCA) 曲线。')
    parser.add_argument('--config', type=str, default='dca_config.json', help='JSON配置文件的路径。')
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 配置文件 '{args.config}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] 配置文件 '{args.config}' 格式不正确。")
        return

    try:
        df = pd.read_csv(config['input_csv_path'])
    except FileNotFoundError:
        print(f"[ERROR] 输入文件 '{config['input_csv_path']}' 未找到。")
        return

    os.makedirs(config.get('output_dir', 'dca_plots'), exist_ok=True)

    for dataset in config['datasets']:
        dataset_name = dataset['name']
        output_filename = f"dca_{dataset_name.replace(' ', '_').lower()}.pdf"
        output_path = os.path.join(config['output_dir'], output_filename)
        
        color = dataset.get('color', '#ed0000')
        
        plot_dca_for_dataset(
            df=df,
            dataset_name=dataset_name,
            models=config['models'],
            threshold_range=config.get('threshold_range', [0, 1]),
            output_path=output_path,
            color=color
        )

    print("\n[INFO] 所有DCA曲线绘制任务已完成。")

if __name__ == "__main__":
    main()