"""
计算单个 Excel 文件中不同 sheet/模型的性能指标。

适用数据格式：
- 一个 Excel 文件；
- 每个 sheet 代表一个模型；
- 每个 sheet 至少包含：ID、label、label_1、group 四列；
- group 列表示该患者属于哪个 cohort，例如 training/internal/external 等；
- label 为真实二分类标签，label_1 为模型预测为 1 类的概率/得分。

安装依赖：
pip install pandas numpy scikit-learn openpyxl

输出指标：
AUC (95% CI) | ACC | Sens | Spec | PPV | NPV | F1

说明：
1. AUC 有 95%CI，使用 bootstrap percentile CI，其他指标都是point_estimate
2. ACC / Sens / Spec / PPV / NPV 输出为百分比形式，保留一位小数，例如 97.0%。
3. F1 输出为小数形式，保留三位小数。
4. 预测标签规则保持：
   y_pred = (y_score >= threshold).astype(int)
5. 当 --model_sheet all 时，会在同一个 summary 表格中汇总输出全部 sheet/模型的 performance。

示例1：输出所有 sheet/模型的 performance_summary
python model_performance_summary_excel_sheets_auc_point_estimate_percent.py ^
  --input_excel "./finaldata/model_predictions.xlsx" ^
  --model_sheet all ^
  --output_path "./summary/all_models_performance_summary.csv"

示例2：只输出某一个模型/sheet的 performance_summary
python model_performance_summary_excel_sheets_auc_point_estimate_percent.py ^
  --input_excel "./finaldata/model_predictions.xlsx" ^
  --model_sheet "uniHEIHC" ^
  --output_path "./summary/uniHEIHC_performance_summary.csv"

示例3：如果你的列名不同，可以手动指定
python model_performance_summary_excel_sheets_auc_point_estimate_percent.py ^
  --input_excel "./finaldata/model_predictions.xlsx" ^
  --model_sheet all ^
  --id_col SampleID ^
  --label_col true_label ^
  --score_col label_1 ^
  --group_col group ^
  --threshold 0.5 ^
  --output_path "./summary/all_models_performance_summary.csv"

实际使用
cd C:/Users/Administrator/Desktop/iScience/performance_summary
python model_performance_summary_excel_sheets_auc_point_estimate_percent.py   --input_excel "模型预测原始结果.xlsx"   --model_sheet all   --output_path "summary.csv"

"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
)


# =========================================================
# Format helpers
# =========================================================
def format_percent(x, digits=1):
    """将0-1之间的小数格式化为百分比字符串，例如0.97 -> 97.0%。"""
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.{digits}f}%"


# =========================================================
# Bootstrap AUC CI
# =========================================================
def bootstrap_auc_ci(
    y_true,
    y_score,
    n_boot=1000,
    ci=95,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        boot_y_true = np.asarray(y_true)[idx]
        boot_y_score = np.asarray(y_score)[idx]

        # AUC要求至少两类
        if len(np.unique(boot_y_true)) < 2:
            continue

        try:
            auc = roc_auc_score(boot_y_true, boot_y_score)
            scores.append(auc)
        except Exception:
            continue

    if len(scores) == 0:
        return np.nan, np.nan

    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)

    return lower, upper


# =========================================================
# Calculate Metrics
# =========================================================
def calculate_metrics(
    y_true,
    y_score,
    threshold=0.5,
    n_boot=1000,
    random_state=2026,
):

    y_true = pd.to_numeric(
        pd.Series(y_true),
        errors="coerce"
    ).to_numpy(dtype=float)

    y_score = pd.to_numeric(
        pd.Series(y_score),
        errors="coerce"
    ).to_numpy(dtype=float)

    valid_idx = ~(np.isnan(y_true) | np.isnan(y_score))

    y_true = y_true[valid_idx].astype(int)
    y_score = y_score[valid_idx].astype(float)

    if len(y_true) < 2:
        return None

    # =====================================================
    # 保持用户要求：>= threshold 判为1
    # =====================================================
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # =====================================================
    # AUC + 95%CI
    # =====================================================
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)

        auc_lower, auc_upper = bootstrap_auc_ci(
            y_true,
            y_score,
            n_boot=n_boot,
            ci=95,
            random_state=random_state,
        )

        auc_ci_str = f"{auc:.3f} ({auc_lower:.3f}-{auc_upper:.3f})"

    else:
        auc_ci_str = "NA"

    # =====================================================
    # Point-estimate metrics
    # =====================================================
    acc = accuracy_score(y_true, y_pred)

    sens = (
        recall_score(y_true, y_pred, zero_division=0)
        if (tp + fn) > 0
        else np.nan
    )

    spec = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    ppv = (
        precision_score(y_true, y_pred, zero_division=0)
        if (tp + fp) > 0
        else np.nan
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else np.nan
    )

    f1 = (
        f1_score(y_true, y_pred, zero_division=0)
        if (tp + fp + fn) > 0
        else np.nan
    )

    return {
        "AUC (95% CI)": auc_ci_str,
        "ACC": format_percent(acc, 1),
        "Sens": format_percent(sens, 1),
        "Spec": format_percent(spec, 1),
        "PPV": format_percent(ppv, 1),
        "NPV": format_percent(npv, 1),
        "F1": round(f1, 3) if not pd.isna(f1) else "NA",
    }


# =========================================================
# Validate Columns
# =========================================================
def validate_columns(df, required_cols, sheet_name):

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' 缺少必要列: {missing}\n"
            f"当前列名:\n{list(df.columns)}"
        )


# =========================================================
# Process One Sheet
# =========================================================
def process_one_sheet(
    df,
    sheet_name,
    id_col,
    label_col,
    score_col,
    group_col,
    threshold,
    n_boot,
    random_state,
):

    validate_columns(
        df,
        [id_col, label_col, score_col, group_col],
        sheet_name
    )

    work = df[
        [id_col, label_col, score_col, group_col]
    ].copy()

    work[label_col] = pd.to_numeric(
        work[label_col],
        errors="coerce"
    )

    work[score_col] = pd.to_numeric(
        work[score_col],
        errors="coerce"
    )

    work[group_col] = (
        work[group_col]
        .astype(str)
        .str.strip()
    )

    work = work.dropna(
        subset=[
            id_col,
            label_col,
            score_col,
            group_col
        ]
    )

    results = []

    cohorts = sorted(
        work[group_col]
        .dropna()
        .unique()
        .tolist()
    )

    for cohort in cohorts:

        sub = work[
            work[group_col] == cohort
        ]

        if len(sub) < 2:
            print(
                f"[Warning] {sheet_name} | {cohort} "
                f"有效样本数 < 2，跳过"
            )
            continue

        metrics = calculate_metrics(
            sub[label_col],
            sub[score_col],
            threshold=threshold,
            n_boot=n_boot,
            random_state=random_state,
        )

        if metrics is None:
            continue

        results.append({
            "模型": sheet_name,
            "数据集": cohort,
            "N": len(sub),
            "Positive_N": int(
                (sub[label_col] == 1).sum()
            ),
            "Negative_N": int(
                (sub[label_col] == 0).sum()
            ),
            **metrics,
        })

        print(
            f"[Info] 已处理: "
            f"{sheet_name} | {cohort} | N={len(sub)}"
        )

    return results


# =========================================================
# Process Excel
# =========================================================
def process_excel(
    input_excel,
    model_sheet,
    output_path,
    id_col,
    label_col,
    score_col,
    group_col,
    threshold,
    n_boot,
    random_state,
):

    if not os.path.exists(input_excel):
        raise FileNotFoundError(
            f"找不到输入文件: {input_excel}"
        )

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    xls = pd.ExcelFile(input_excel)

    sheet_names = xls.sheet_names

    if model_sheet.lower() == "all":
        selected_sheets = sheet_names
    else:

        if model_sheet not in sheet_names:
            raise ValueError(
                f"指定sheet不存在: {model_sheet}\n"
                f"可用sheet:\n{sheet_names}"
            )

        selected_sheets = [model_sheet]

    print(f"[Info] 输入Excel: {input_excel}")
    print(f"[Info] 处理sheet: {selected_sheets}")

    all_results = []

    for sheet in selected_sheets:

        df = pd.read_excel(
            input_excel,
            sheet_name=sheet
        )

        results = process_one_sheet(
            df=df,
            sheet_name=sheet,
            id_col=id_col,
            label_col=label_col,
            score_col=score_col,
            group_col=group_col,
            threshold=threshold,
            n_boot=n_boot,
            random_state=random_state,
        )

        all_results.extend(results)

    if len(all_results) == 0:
        raise RuntimeError(
            "未得到任何有效结果"
        )

    result_df = pd.DataFrame(all_results)

    # =====================================================
    # Column Order
    # =====================================================
    col_order = [
        "模型",
        "数据集",
        "N",
        "Positive_N",
        "Negative_N",
        "AUC (95% CI)",
        "ACC",
        "Sens",
        "Spec",
        "PPV",
        "NPV",
        "F1",
    ]

    result_df = result_df[col_order]

    # =====================================================
    # Save
    # =====================================================
    ext = os.path.splitext(
        output_path
    )[1].lower()

    if ext in [".xlsx", ".xls"]:
        result_df.to_excel(
            output_path,
            index=False
        )
    else:
        result_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

    print(
        f"\n[Success] 结果已保存到:\n{output_path}"
    )


# =========================================================
# Main
# =========================================================
def main():

    parser = argparse.ArgumentParser(
        description=(
            "计算Excel不同sheet模型在不同cohort中的性能指标"
        )
    )

    parser.add_argument(
        "--input_excel",
        required=True,
        help="输入Excel路径"
    )

    parser.add_argument(
        "--model_sheet",
        required=True,
        help="sheet名称；全部模型填写 all"
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="输出路径(.csv/.xlsx)"
    )

    parser.add_argument(
        "--id_col",
        default="SampleID",
        help="患者ID列名"
    )

    parser.add_argument(
        "--label_col",
        default="true_label",
        help="真实标签列名"
    )

    parser.add_argument(
        "--score_col",
        default="label_1",
        help="预测概率列名"
    )

    parser.add_argument(
        "--group_col",
        default="group",
        help="cohort列名"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="二分类阈值"
    )

    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="bootstrap次数"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=2026,
        help="随机种子"
    )

    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        process_excel(
            input_excel=args.input_excel,
            model_sheet=args.model_sheet,
            output_path=args.output_path,
            id_col=args.id_col,
            label_col=args.label_col,
            score_col=args.score_col,
            group_col=args.group_col,
            threshold=args.threshold,
            n_boot=args.n_boot,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    print("当前运行的是 percent 版本脚本")
    main()
