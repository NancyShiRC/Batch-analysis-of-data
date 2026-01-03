"""
cd C:/Users/Administrator/Desktop/CRLM

python calibration_HL_Brier.py     --intotal_path SRC-table/intotal-cbind.csv   --config_path SRC-calibration/calibration_HL_Brier.json   --output_path SRC-calibration/calibration_HL_Brier.csv"


"""
import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from scipy.stats import chi2
import traceback

class Color:
    RESET = '\033[0m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        required_keys = ['models', 'positive_label']
        if not all(key in config for key in required_keys):
            raise ValueError(f"配置文件缺少必要的键: {', '.join(required_keys)}")
        
        config.setdefault('Hosmer-Lemeshow_groups', 10)  
        config.setdefault('datasets', ['all'])  
        if config['datasets'] == ['all']:
            print(f"{Color.YELLOW}[Warning] 未指定数据集，将计算所有数据的结果。{Color.RESET}")
        
        return config
    except FileNotFoundError:
        print(f"{Color.RED}[Error] 配置文件未找到: {config_path}{Color.RESET}")
        raise
    except json.JSONDecodeError:
        print(f"{Color.RED}[Error] 配置文件格式错误（非有效JSON）。{Color.RESET}")
        raise
    except Exception as e:
        print(f"{Color.RED}[Error] 加载配置文件失败: {e}{Color.RESET}")
        raise

def hosmer_lemeshow_test(y_true, y_pred, groups=10):
    if len(y_true) < groups * 2:
        print(f"{Color.YELLOW}[Warning] 数据量不足（{len(y_true)}条），无法按{groups}组执行HL检验，返回NaN。{Color.RESET}")
        return np.nan, np.nan
    
    data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    try:
        data['decile'] = pd.qcut(data['y_pred'], groups, labels=False, duplicates='drop')
        actual_groups = data['decile'].nunique()
        if actual_groups < groups:
            print(f"{Color.YELLOW}[Warning] 预测概率重复值较多，实际分组数{actual_groups}（小于指定{groups}组）。{Color.RESET}")
    except ValueError:
        print(f"{Color.RED}[Error] 预测概率全部相同，无法分组，返回NaN。{Color.RESET}")
        return np.nan, np.nan
    
    observed = data.groupby('decile')['y_true'].agg(
        observed_events='sum',
        observed_non_events=lambda x: (x == 0).sum()
    )
    expected = data.groupby('decile').agg(
        expected_events=('y_pred', 'sum'),
        expected_non_events=('y_pred', lambda x: len(x) - x.sum())
    )
    contingency = pd.concat([observed, expected], axis=1)
    
    contingency = contingency.replace(0, 1e-10)
    
    hl_chi2 = np.sum(
        (contingency['observed_events'] - contingency['expected_events'])**2 / contingency['expected_events'] +
        (contingency['observed_non_events'] - contingency['expected_non_events'])**2 / contingency['expected_non_events']
    )
    
    dof = actual_groups - 2
    if dof <= 0:
        print(f"{Color.RED}[Error] 自由度={dof}（需>0），无法计算P值，返回NaN。{Color.RESET}")
        return hl_chi2, np.nan
    hl_p = 1 - chi2.cdf(hl_chi2, dof)
    
    return hl_chi2, hl_p

def calculate_metrics_by_dataset(intotal_df: pd.DataFrame, model_col: str, pos_label: int, dataset: str, hl_groups: int) -> dict:

    if dataset != 'all':
        data_filtered = intotal_df[intotal_df['set'] == dataset].copy()
        if len(data_filtered) == 0:
            print(f"{Color.YELLOW}[Warning] 数据集'{dataset}'无数据，模型{model_col}跳过该数据集。{Color.RESET}")
            return None
    else:
        data_filtered = intotal_df.copy()
    

    valid_mask = data_filtered['true_label'].notna() & data_filtered[model_col].notna()
    y_true = data_filtered.loc[valid_mask, 'true_label']
    y_pred_prob = data_filtered.loc[valid_mask, model_col]
    
    if len(y_true) < 5: 
        print(f"{Color.YELLOW}[Warning] 模型{model_col}在数据集'{dataset}'有效数据仅{len(y_true)}条，跳过。{Color.RESET}")
        return None
    

    y_true_binary = (y_true == pos_label).astype(int)
    
    brier = round(brier_score_loss(y_true_binary, y_pred_prob), 4)
    
    hl_chi2, hl_p = hosmer_lemeshow_test(y_true_binary, y_pred_prob, hl_groups)
    hl_chi2 = round(hl_chi2, 4) if not np.isnan(hl_chi2) else np.nan
    hl_p = round(hl_p, 4) if not np.isnan(hl_p) else np.nan
    
    return {
        'Model': model_col,
        'Dataset': dataset,
        'Sample_Count': len(y_true),  
        'Brier_Score': brier,
        'HL_Chi2': hl_chi2,
        'HL_p_value': hl_p
    }

def main():
    parser = argparse.ArgumentParser(
        description='按数据集计算模型的Brier Score和Hosmer-Lemeshow检验',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--intotal_path', type=str, required=True, help='intotal.csv文件路径')
    parser.add_argument('--config_path', type=str, required=True, help='JSON配置文件路径（含models/datasets等）')
    parser.add_argument('--output_path', type=str, required=True, help='输出结果CSV路径')
    args = parser.parse_args()

    try:
        config = load_config(args.config_path)
        models = config['models']
        datasets = config['datasets']
        pos_label = config['positive_label']
        hl_groups = config['Hosmer-Lemeshow_groups']
        print(f"{Color.BLUE}[Info] 配置加载完成：模型{len(models)}个，数据集{datasets}，正标签{pos_label}。{Color.RESET}")

        if not os.path.exists(args.intotal_path):
            raise FileNotFoundError(f"intotal.csv未找到：{args.intotal_path}")
        intotal_df = pd.read_csv(args.intotal_path)
        required_cols = ['set', 'true_label'] + models
        missing_cols = [col for col in required_cols if col not in intotal_df.columns]
        if missing_cols:
            raise ValueError(f"intotal.csv缺少必要列：{', '.join(missing_cols)}")
        print(f"{Color.BLUE}[Info] intotal.csv加载完成，共{len(intotal_df)}条数据。{Color.RESET}")

        all_results = []
        print(f"\n{Color.BOLD}[Info] 开始计算指标...{Color.RESET}")
        for model in models:
            print(f"\n{Color.BLUE}[Info] 正在处理模型：{model}{Color.RESET}")
            for dataset in datasets:
                metrics = calculate_metrics_by_dataset(
                    intotal_df=intotal_df,
                    model_col=model,
                    pos_label=pos_label,
                    dataset=dataset,
                    hl_groups=hl_groups
                )
                if metrics:
                    all_results.append(metrics)

        if not all_results:
            print(f"{Color.YELLOW}[Warning] 未生成任何结果，可能无有效数据。{Color.RESET}")
            return
        
        result_df = pd.DataFrame(all_results)
        result_df = result_df[['Model', 'Dataset', 'Sample_Count', 'Brier_Score', 'HL_Chi2', 'HL_p_value']]
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        result_df.to_csv(args.output_path, index=False, encoding='utf-8-sig')

        print(f"\n{Color.GREEN}[Success] 计算完成！结果已保存至：{args.output_path}{Color.RESET}")
        print(f"\n{Color.BOLD}结果预览：{Color.RESET}")
        print(result_df.to_string(index=False, max_rows=10))   

    except Exception as e:
        print(f"\n{Color.RED}[Error] 程序执行失败：{e}{Color.RESET}")
        traceback.print_exc()

if __name__ == "__main__":
    main()