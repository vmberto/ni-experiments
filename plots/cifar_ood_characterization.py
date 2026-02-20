#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Out-of-Distribution Characterization Pipeline
---------------------------------------------
Generates:
  1. Latent-space divergence curves (CIFAR-10-C & CIFAR-100-C)
  2. Cross-dataset correlation analysis
  3. Intensity-progression trend plots
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

from lib.helpers import seaborn_styles

# ============================================================
# CONFIGURATION
# ============================================================

seaborn_styles(sns)
rcParams.update({
    'font.size': 22,
    'axes.labelsize': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 18,
    'axes.titlesize': 26
})

DATASETS = ["cifar10", "cifar100"]
RESULTS_DIR = "../results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "ood-characterization")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bootstrap_ci(data, n_bootstrap=1000):
    """Compute bootstrap mean and 95% confidence interval."""
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True))
                  for _ in range(n_bootstrap)]
    return np.mean(boot_means), np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def categorize_by_percentiles(pct_value):
    """Map percentile to severity band."""
    if pct_value <= 33:
        return "Lowest"
    elif pct_value <= 66:
        return "Mid-Range"
    else:
        return "Highest"


def load_raw_dataset(dataset):
    """Load raw full-characterization CSV and extract base corruption + intensity."""
    file_path = f"{RESULTS_DIR}/{dataset}/{dataset}_{dataset}c_full_characterization.csv"
    df = pd.read_csv(file_path)
    parts = df["corruption_type"].str.rsplit("_", n=1, expand=True)
    df["base"] = parts[0]
    df["intensity"] = parts[1].astype(int)
    return df


def save_dataframe(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")


# ============================================================
# 1. LATENT-SPACE CURVE CHARACTERIZATION
# ============================================================

def plot_latent_space_trends():
    fig, axes = plt.subplots(1, 2, figsize=(28, 10), sharey=True)

    for i, dataset in enumerate(DATASETS):
        ax = axes[i]
        file_path = f"{RESULTS_DIR}/{dataset}/{dataset}_{dataset}c_full_characterization.csv"
        data = pd.read_csv(file_path)

        # Bootstrap per corruption
        boot = data.groupby("corruption_type")["wasserstein_per_feature"].apply(
            lambda x: bootstrap_ci(x.values))
        grouped = pd.DataFrame(boot.tolist(), index=boot.index,
                               columns=["mean", "lower", "upper"]).reset_index()

        # Add percentile ranks and severity categories
        grouped["percentile"] = grouped["mean"].rank(pct=True) * 100
        grouped["category"] = grouped["percentile"].apply(categorize_by_percentiles)

        # Aggregate stats per severity band
        cat_stats = grouped.groupby("category")["mean"].apply(lambda x: bootstrap_ci(x.values))
        cat_stats = pd.DataFrame(cat_stats.tolist(), index=cat_stats.index,
                                 columns=["cat_avg_mean", "cat_avg_lower", "cat_avg_upper"])

        final_df = grouped.merge(cat_stats, left_on="category", right_index=True)
        save_dataframe(final_df, f"{RESULTS_DIR}/{dataset}/{dataset}_final_characterization_with_stats.csv")

        # Plot
        sns.lineplot(x="percentile", y="mean", data=final_df, color="purple",
                     linewidth=2.5, alpha=0.6, label=f"{dataset.upper()} Trend", ax=ax)
        sns.scatterplot(x="percentile", y="mean", data=final_df,
                        color="red", s=110, edgecolor="white", ax=ax)

        # Error bars
        for _, r in final_df.iterrows():
            ax.errorbar(r["percentile"], r["mean"],
                        yerr=[[r["mean"] - r["lower"]], [r["upper"] - r["mean"]]],
                        fmt="none", ecolor="black", capsize=3, alpha=0.15)

        # Decorations
        ax.scatter(0, 0, color="blue", s=200, zorder=5)
        ax.text(0, -0.05, f"{dataset.upper()}", ha="center", va="top", color="blue",
                fontweight="bold", fontsize=32, transform=ax.get_xaxis_transform())

        for cat, pos in zip(["Lowest", "Mid-Range", "Highest"], [16.5, 49.5, 83]):
            count = (final_df["category"] == cat).sum()
            m, l, u = cat_stats.loc[cat]
            ax.text(pos, 0.92, f"{cat}", ha="center", va="bottom", fontsize=32, fontweight="bold",
                    transform=ax.get_xaxis_transform())
            ax.text(pos, 0.90, f"{count} types\n{m:.3f} ({l:.3f}, {u:.3f})",
                    ha="center", va="top", fontsize=20, transform=ax.get_xaxis_transform())

        for x_val in [33, 66]:
            ax.axvline(x=x_val, color="black", linestyle="-", linewidth=1.2, alpha=0.3)

        ax.set_xlim(0, 105)
        ax.set_ylim(0, 0.14)
        ax.set_xlabel("Percentiles of Corruption Severity", fontsize=28)
        if i == 0:
            ax.set_ylabel("Wasserstein Distance", fontsize=32)
        ax.set_xticks([0, 33, 66, 100])

        ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02), frameon=True,
                  facecolor="white", edgecolor="black", framealpha=0.8)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "combined_wasserstein_characterization.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved plot: {out_path}")


# ============================================================
# 2. CROSS-DATASET CORRELATION ANALYSIS
# ============================================================

def plot_cross_dataset_correlation():
    df10 = pd.read_csv(f"{RESULTS_DIR}/cifar10/cifar10_final_characterization_with_stats.csv")
    df100 = pd.read_csv(f"{RESULTS_DIR}/cifar100/cifar100_final_characterization_with_stats.csv")

    merged = pd.merge(df10[["corruption_type", "mean"]],
                      df100[["corruption_type", "mean"]],
                      on="corruption_type", suffixes=("_c10", "_c100"))

    pearson_r = merged["mean_c10"].corr(merged["mean_c100"], method="pearson")
    spearman_r = merged["mean_c10"].corr(merged["mean_c100"], method="spearman")

    print(f"[INFO] CIFAR-10 vs CIFAR-100 correlation:")
    print(f"  Pearson r = {pearson_r:.3f}")
    print(f"  Spearman r = {spearman_r:.3f}")

    plt.figure(figsize=(10, 8))
    sns.regplot(data=merged, x="mean_c10", y="mean_c100",
                scatter_kws={"s": 100, "alpha": 0.7, "edgecolor": "white"},
                line_kws={"color": "red", "linewidth": 2.5})
    plt.xlabel("CIFAR-10-C Mean Wasserstein Distance", fontsize=24)
    plt.ylabel("CIFAR-100-C Mean Wasserstein Distance", fontsize=24)
    plt.title(f"CIFAR-10-C vs CIFAR-100-C Correlation\n"
              f"Pearson r={pearson_r:.3f}, Spearman r={spearman_r:.3f}",
              fontsize=22, pad=20)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "correlation_cifar10_vs_cifar100.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved plot: {out_path}")


# ============================================================
# 4. OUTLIER DETECTION: NON-MONOTONIC INTENSITY BEHAVIOR
# ============================================================

def detect_non_monotonic_trends():
    print("\n[INFO] Detecting non-monotonic corruption trends...")

    results = []
    for dataset in DATASETS:
        df = load_raw_dataset(dataset)
        summary = df.groupby(["base", "intensity"])["wasserstein_per_feature"].mean().reset_index()

        for base, group in summary.groupby("base"):
            vals = group.sort_values("intensity")["wasserstein_per_feature"].values
            diffs = np.diff(vals)
            # A violation occurs when difference < 0 (i.e., decreases when it should increase)
            violations = (diffs < 0).sum()
            total = len(diffs)
            violation_ratio = violations / total if total > 0 else 0

            if violations > 0:
                results.append({
                    "dataset": dataset,
                    "corruption": base,
                    "values": np.round(vals, 5).tolist(),
                    "violations": int(violations),
                    "violation_ratio": violation_ratio
                })

    outliers_df = pd.DataFrame(results).sort_values(["violations", "violation_ratio"], ascending=[False, False])
    out_path = os.path.join(PLOTS_DIR, "non_monotonic_outliers.csv")
    save_dataframe(outliers_df, out_path)

    print(f"⚠️ Found {len(outliers_df)} non-monotonic corruption patterns.")
    if not outliers_df.empty:
        print(outliers_df.head(10)[["dataset", "corruption", "values", "violations"]])
    else:
        print("✅ All corruption families show monotonic progression.")

# ============================================================
# 6. COMBINED 2x2 FIGURE (Intensity + Non-Monotonic — Final Clean Version)
# ============================================================

def plot_combined_intensity_and_nonmonotonic():
    print("\n[INFO] Generating final 2×2 intensity/non-monotonic figure (clean, no CI fill)...")

    # Load data
    df10 = load_raw_dataset("cifar10")
    df100 = load_raw_dataset("cifar100")
    outliers_path = os.path.join(PLOTS_DIR, "non_monotonic_outliers.csv")
    outliers_df = pd.read_csv(outliers_path) if os.path.exists(outliers_path) else pd.DataFrame()

    # Helper: format captions (snake_case → Capitalized Words)
    def format_caption(name):
        return name.replace("_", " ").title()

    # Helper: compute mean per corruption + intensity
    def summarize(df):
        return df.groupby(["base", "intensity"])["wasserstein_per_feature"].agg(["mean", "std"]).reset_index()

    sum10, sum100 = summarize(df10), summarize(df100)

    # Top-row: representative corruption types for monotonic illustration
    top_corruptions = ["brightness", "contrast", "fog", "gaussian_noise"]

    # Bottom-row: non-monotonic families detected previously
    nonmono10 = outliers_df[outliers_df["dataset"] == "cifar10"]["corruption"].unique().tolist() if not outliers_df.empty else []
    nonmono100 = outliers_df[outliers_df["dataset"] == "cifar100"]["corruption"].unique().tolist() if not outliers_df.empty else []
    nonmono10 = [c for c in nonmono10 if c != "motion_blur"]
    nonmono100 = [c for c in nonmono100 if c != "motion_blur"]

    # Create figure layout
    fig, axes = plt.subplots(2, 2, figsize=(26, 16), sharey=True)

    # --- TOP ROW: regular intensity progression ---
    for index, (ax, (dataset, summary, corrs)) in enumerate(zip(
        axes[0],
        [
            ("CIFAR-10-C", sum10, top_corruptions),
            ("CIFAR-100-C", sum100, top_corruptions),
        ],
    )):
        subset = summary[summary["base"].isin(corrs)].copy()
        subset["base"] = subset["base"].apply(format_caption)

        ax.set_xticks([1, 2, 3, 4, 5])

        sns.lineplot(
            data=subset,
            x="intensity", y="mean", hue="base", linewidth=3, marker="o", ax=ax
        )
        ax.set_title(f"{dataset}: Intensity Progression", fontsize=34)
        ax.set_xlabel("", fontsize=32)
        ax.set_ylabel("Mean Wasserstein Distance", fontsize=34)
        ax.legend(title="", fontsize=24, title_fontsize=24)
        ax.grid(alpha=0.3)

    # --- BOTTOM ROW: clean non-monotonic visualization (no CI fill, same Y-scale) ---
    for ax, (dataset, df, nonmono) in zip(
            axes[1],
            [
                ("CIFAR-10-C", df10, nonmono10),
                ("CIFAR-100-C", df100, nonmono100),
            ],
    ):
        subset = df[df["base"].isin(nonmono)].copy()
        if subset.empty:
            ax.text(0.5, 0.5, "No non-monotonic patterns found", fontsize=22,
                    ha="center", va="center", color="gray", transform=ax.transAxes)
            continue

        # 🔹 Aggregate to match top plots
        subset_summary = (
            subset.groupby(["base", "intensity"])["wasserstein_per_feature"]
            .mean().reset_index()
        )
        subset_summary["base"] = subset_summary["base"].apply(format_caption)

        sns.lineplot(
            data=subset_summary,
            x="intensity", y="wasserstein_per_feature",
            hue="base", style="base",
            markers=True, dashes=False,
            linewidth=3, marker="o", ax=ax,
            errorbar=None
        )

        # 🔹 Force same Y range as top plots
        ax.set_xticks([1, 2, 3, 4, 5])

        ax.set_title(f"{dataset}: Non-Monotonic Trends", fontsize=34)
        ax.set_xlabel("Intensity Level (1–5)", fontsize=34)
        ax.set_ylabel("Mean Wasserstein Distance", fontsize=34)
        ax.legend(title="", fontsize=24, title_fontsize=24)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "combined_intensity_nonmonotonic_clean.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved clean combined figure (no CI fill): {out_path}")


# ============================================================
# MAIN EXECUTION PIPELINE
# ============================================================
if __name__ == "__main__":
    print("=== Out-of-Distribution Characterization Pipeline ===")
    plot_latent_space_trends()
    plot_cross_dataset_correlation()
    detect_non_monotonic_trends()
    plot_combined_intensity_and_nonmonotonic()

    print("✅ All analyses completed successfully.")
