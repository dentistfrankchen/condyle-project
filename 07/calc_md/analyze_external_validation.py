"""
External Validation Set Analysis Script
Computes diagnostic performance metrics for the external validation cohort (70 cases),
using balanced accuracy and its bootstrap 95% CI.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import itertools

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / 'external_validation_data.csv'

def bootstrap_balanced_accuracy_ci(y_true, y_score, n_bootstrap=2000, alpha=0.95, random_state=42):
    """Compute bootstrap CI for balanced accuracy."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)

    stats_list = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = np.rint(y_score[idx]).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())

        pos = tp + fn
        neg = tn + fp
        if pos == 0 or neg == 0:
            continue

        sens = tp / pos
        spec = tn / neg
        stats_list.append(0.5 * (sens + spec))

    if not stats_list:
        return np.nan, np.nan

    lower = np.percentile(stats_list, (1 - alpha) / 2 * 100)
    upper = np.percentile(stats_list, (1 + alpha) / 2 * 100)
    return float(lower), float(upper)

def calculate_confidence_interval(proportion, n, confidence=0.95):
    """Compute proportion CI (Wilson Score Interval)."""
    if n == 0:
        return 0, 0
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((proportion * (1 - proportion) / n + z**2 / (4 * n**2))) / denominator
    return center - margin, center + margin

def model_metrics(df):
    """Compute diagnostic performance metrics for each model."""
    models = ['Gemini-3', 'GPT-5.2', 'Qwen3']
    rows = []
    
    for model_name in models:
        y_true = df['y_true'].values
        y_pred = df[model_name].values

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        acc = (tp + tn) / len(y_true)
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan

        ba_val = (sens + spec) / 2
        ci_low, ci_high = bootstrap_balanced_accuracy_ci(y_true, y_pred)

        rows.append(
            {
                'Model': model_name,
                'N': len(y_true),
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'Accuracy': round(acc, 4),
                'Sensitivity': round(sens, 4),
                'Specificity': round(spec, 4),
                'Balanced_Accuracy': round(float(ba_val), 4),
                'Balanced_Accuracy_95CI_L': round(ci_low, 4),
                'Balanced_Accuracy_95CI_U': round(ci_high, 4),
            }
        )
    
    return pd.DataFrame(rows)

def mcnemar_exact_p(b, c):
    """Compute exact McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(stats.binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue)

def pairwise_mcnemar(df):
    """Pairwise McNemar test for all model pairs."""
    models = ['Gemini-3', 'GPT-5.2', 'Qwen3']
    rows = []
    
    pairs = list(itertools.combinations(models, 2))
    for m1, m2 in pairs:
        c1 = (df[m1].values == df['y_true'].values)
        c2 = (df[m2].values == df['y_true'].values)

        a = int((c1 & c2).sum())
        b = int((c1 & ~c2).sum())
        c = int((~c1 & c2).sum())
        d = int((~c1 & ~c2).sum())
        p = mcnemar_exact_p(b, c)

        rows.append(
            {
                'Model_1': m1,
                'Model_2': m2,
                'Both_Correct': a,
                'M1_Correct_M2_Wrong': b,
                'M1_Wrong_M2_Correct': c,
                'Both_Wrong': d,
                'Discordant': b + c,
                'McNemar_Exact_P': p,
                'Significant_p<0.05': 'Yes' if p < 0.05 else 'No',
            }
        )
    
    return pd.DataFrame(rows)

def main():
    if not DATA_FILE.exists():
        print(f"Data file not found: {DATA_FILE}")
        print("Please provide external_validation_data.csv")
        return
    
    df = pd.read_csv(DATA_FILE)
    
    print("\n" + "="*80)
    print("External Validation Set Analysis")
    print("="*80)
    
    metrics_df = model_metrics(df)
    mcnemar_df = pairwise_mcnemar(df)
    
    metrics_output = ROOT / 'external_validation_model_performance.csv'
    mcnemar_output = ROOT / 'external_validation_mcnemar_results.csv'
    
    metrics_df.to_csv(metrics_output, index=False, encoding='utf-8-sig')
    mcnemar_df.to_csv(mcnemar_output, index=False, encoding='utf-8-sig')
    
    print("\nModel Performance (Balanced Accuracy with 95% Bootstrap CI):")
    print(metrics_df.to_string(index=False))
    
    print("\n\nPairwise McNemar Test Results:")
    print(mcnemar_df.to_string(index=False))
    
    print(f"\n\nResults saved:")
    print(f"  - {metrics_output}")
    print(f"  - {mcnemar_output}")
    
    # Cross-validation comparison
    print("\n\nInternal vs External Validation Comparison:")
    print("="*80)
    
    internal_metrics = pd.read_csv(ROOT / 'Model_Performance_with_AUC_CI.csv')
    
    comparison = pd.DataFrame({
        'Model': metrics_df['Model'],
        'Internal_Accuracy': internal_metrics['Accuracy'].values,
        'External_Accuracy': metrics_df['Accuracy'].values,
        'Internal_Balanced_Accuracy': internal_metrics['Balanced_Accuracy'].values,
        'External_Balanced_Accuracy': metrics_df['Balanced_Accuracy'].values,
        'Internal_Balanced_Accuracy_CI': [f"{row['Balanced_Accuracy_95CI_L']:.3f}-{row['Balanced_Accuracy_95CI_U']:.3f}" for _, row in internal_metrics.iterrows()],
        'External_Balanced_Accuracy_CI': [f"{row['Balanced_Accuracy_95CI_L']:.3f}-{row['Balanced_Accuracy_95CI_U']:.3f}" for _, row in metrics_df.iterrows()],
    })
    
    print(comparison.to_string(index=False))
    
    comparison.to_csv(ROOT / 'validation_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"  - {ROOT / 'validation_comparison.csv'}")

if __name__ == '__main__':
    main()
