import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]
ABNORMAL_DIR = ROOT / "abnormal_cases"
NORMAL_DIR_CANDIDATES = [
    ROOT / "normal_cases" / "gemini-gpt-qwen-send",
    ROOT / "normal_cases" / "gemini-gpt-qwen",
]
OUT_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Gemini-3": "results_openrouter_gemini_pro_no_prompt.csv",
    "GPT-5.2": "results_openrouter_gpt5_2.csv",
    "Qwen3": "results_openrouter_qwen3.csv",
}


def pick_normal_dir() -> Path:
    for d in NORMAL_DIR_CANDIDATES:
        if d.exists():
            return d
    raise FileNotFoundError("Normal cases directory not found")


def to_binary(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer, float, np.floating)):
        return int(round(float(x)))

    s = str(x).strip().lower()
    if s in {"1", "yes", "y", "true", "positive", "pos"}:
        return 1
    if s in {"0", "no", "n", "false", "negative", "neg"}:
        return 0
    return np.nan


def load_model_predictions(model_name: str, normal_dir: Path):
    file_name = MODEL_FILES[model_name]
    abnormal_path = ABNORMAL_DIR / file_name
    normal_path = normal_dir / file_name

    if not abnormal_path.exists():
        raise FileNotFoundError(f"Abnormal file not found: {abnormal_path}")
    if not normal_path.exists():
        raise FileNotFoundError(f"Normal file not found: {normal_path}")

    abn = pd.read_csv(abnormal_path)
    nor = pd.read_csv(normal_path)

    abn = abn[["filename", "Osteophyte_Formation"]].copy()
    nor = nor[["filename", "Osteophyte_Formation"]].copy()

    abn["y_true"] = 1
    nor["y_true"] = 0
    abn["split"] = "abnormal"
    nor["split"] = "normal"

    df = pd.concat([abn, nor], ignore_index=True)
    df[model_name] = df["Osteophyte_Formation"].apply(to_binary)
    df = df.drop(columns=["Osteophyte_Formation"])

    return df


def merge_all_models(normal_dir: Path):
    merged = None
    for model_name in MODEL_FILES:
        df_m = load_model_predictions(model_name, normal_dir)
        if merged is None:
            merged = df_m
        else:
            merged = merged.merge(df_m[["filename", "split", model_name]], on=["filename", "split"], how="inner")

    merged = merged.dropna(subset=list(MODEL_FILES.keys()))
    for m in MODEL_FILES:
        merged[m] = merged[m].astype(int)

    return merged


def bootstrap_balanced_accuracy_ci(y_true, y_score, n_bootstrap=2000, alpha=0.95, random_state=42):
    """Bootstrap CI for balanced accuracy."""
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


def model_metrics(df):
    rows = []
    for model_name in MODEL_FILES:
        y_true = df["y_true"].values
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
                "Model": model_name,
                "N": len(y_true),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Accuracy": round(acc, 4),
                "Sensitivity": round(sens, 4),
                "Specificity": round(spec, 4),
                "Balanced_Accuracy": round(float(ba_val), 4),
                "Balanced_Accuracy_95CI_L": round(ci_low, 4),
                "Balanced_Accuracy_95CI_U": round(ci_high, 4),
            }
        )
    return pd.DataFrame(rows)


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue)


def pairwise_mcnemar(df):
    rows = []
    pairs = list(itertools.combinations(MODEL_FILES.keys(), 2))
    for m1, m2 in pairs:
        c1 = (df[m1].values == df["y_true"].values)
        c2 = (df[m2].values == df["y_true"].values)

        a = int((c1 & c2).sum())
        b = int((c1 & ~c2).sum())
        c = int((~c1 & c2).sum())
        d = int((~c1 & ~c2).sum())
        p = mcnemar_exact_p(b, c)

        rows.append(
            {
                "Model_1": m1,
                "Model_2": m2,
                "Both_Correct": a,
                "M1_Correct_M2_Wrong": b,
                "M1_Wrong_M2_Correct": c,
                "Both_Wrong": d,
                "Discordant": b + c,
                "McNemar_Exact_P": p,
                "Significant_p<0.05": "Yes" if p < 0.05 else "No",
            }
        )
    return pd.DataFrame(rows)


def plot_mcnemar_tables(mcnemar_df: pd.DataFrame, output_path: Path):
    n = len(mcnemar_df)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), dpi=160)
    if n == 1:
        axes = [axes]

    for ax, row in zip(axes, mcnemar_df.to_dict(orient="records")):
        table = np.array(
            [
                [row["Both_Correct"], row["M1_Correct_M2_Wrong"]],
                [row["M1_Wrong_M2_Correct"], row["Both_Wrong"]],
            ]
        )
        im = ax.imshow(table, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"{row['Model_2']} Correct", f"{row['Model_2']} Wrong"], rotation=20, ha="right")
        ax.set_yticklabels([f"{row['Model_1']} Correct", f"{row['Model_1']} Wrong"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(table[i, j]), ha="center", va="center", color="black", fontsize=11)

        ax.set_title(f"{row['Model_1']} vs {row['Model_2']}\nMcNemar p={row['McNemar_Exact_P']:.4g}")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Pairwise McNemar Contingency Tables", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    normal_dir = pick_normal_dir()
    df = merge_all_models(normal_dir)

    metrics_df = model_metrics(df)
    mcnemar_df = pairwise_mcnemar(df)

    metrics_path = OUT_DIR / "Model_Performance_with_AUC_CI.csv"
    mcnemar_path = OUT_DIR / "Pairwise_McNemar_Results.csv"
    mcnemar_fig_path = OUT_DIR / "Pairwise_McNemar_Contingency_Tables.png"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    mcnemar_df.to_csv(mcnemar_path, index=False, encoding="utf-8-sig")

    plot_mcnemar_tables(mcnemar_df, mcnemar_fig_path)

    print("Analysis complete. Output files:")
    print(f"- {metrics_path}")
    print(f"- {mcnemar_path}")
    print(f"- {mcnemar_fig_path}")

    print("\nModel Performance (Balanced Accuracy with 95% CI):")
    print(metrics_df.to_string(index=False))

    print("\nPairwise McNemar Test Results:")
    print(mcnemar_df.to_string(index=False))


if __name__ == "__main__":
    main()

