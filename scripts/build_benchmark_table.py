"""Build a benchmark table from already-committed per-drug CV AUC POC results.

Reads ``results/poc/per_drug_cv_auc.csv`` (two feature sets per drug: DDR+MSI
alone vs DDR+MSI+Tissue context) and writes a markdown-formatted benchmark
table contrasting the two. Additive-only: does not modify ``results/poc/*``.

Run:
    python scripts/build_benchmark_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
POC_PATH = REPO_ROOT / "results" / "poc" / "per_drug_cv_auc.csv"
OUT_DIR = REPO_ROOT / "results" / "benchmark"
OUT_PATH = OUT_DIR / "benchmark_table.md"


def build() -> str:
    df = pd.read_csv(POC_PATH)
    pivot = df.pivot_table(
        index=["drug_id", "drug_name", "n", "n_sensitive"],
        columns="feature_set",
        values=["cv_auc_mean", "cv_auc_std"],
    ).reset_index()
    pivot.columns = ["_".join(c).strip("_") for c in pivot.columns.to_flat_index()]

    lines = [
        "# Benchmark: DDR+MSI alone vs DDR+MSI + Tissue context",
        "",
        "Five-fold stratified CV AUC per drug, two feature sets:",
        "1. `DDR + MSI_FACTOR` — mutation + MSI only (12 features, no tissue)",
        "2. `DDR + MSI + Tissue` — with tissue one-hot (39 features)",
        "",
        "Source: `results/poc/per_drug_cv_auc.csv` (committed POC v2).",
        "",
        "| Drug | n | n_sens | AUC (DDR+MSI) | AUC (+Tissue) | Δ AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in pivot.iterrows():
        base = row["cv_auc_mean_DDR + MSI_FACTOR"]
        full = row["cv_auc_mean_DDR + MSI + Tissue"]
        base_sd = row["cv_auc_std_DDR + MSI_FACTOR"]
        full_sd = row["cv_auc_std_DDR + MSI + Tissue"]
        delta = full - base
        lines.append(
            f"| {row['drug_name']} | {int(row['n'])} | {int(row['n_sensitive'])} "
            f"| {base:.3f} ± {base_sd:.3f} | {full:.3f} ± {full_sd:.3f} "
            f"| +{delta:.3f} |"
        )

    mean_base = pivot["cv_auc_mean_DDR + MSI_FACTOR"].mean()
    mean_full = pivot["cv_auc_mean_DDR + MSI + Tissue"].mean()
    lines += [
        "",
        f"**Mean AUC (DDR+MSI only): {mean_base:.3f}**",
        f"**Mean AUC (+Tissue): {mean_full:.3f}**",
        f"**Mean Δ from tissue context: +{mean_full - mean_base:.3f} AUC**",
        "",
        "## Interpretation",
        "",
        "DDR + MSI features alone discriminate sensitivity barely above chance",
        "(mean AUC ≈ 0.58). Adding tissue-of-origin one-hots lifts mean AUC to",
        "≈ 0.73, i.e. tissue context contributes ~0.15 AUC — the DDR/MSI signal",
        "on cell lines is real but modest, and the headline AUC numbers in the",
        "portfolio site reflect the full tissue-aware model. This is the",
        "honest-negative framing called out in the README's Limits section.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build())
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
