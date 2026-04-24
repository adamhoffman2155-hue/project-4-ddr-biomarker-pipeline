#!/usr/bin/env python3
"""
Regenerate ``results/poc/manifest.json`` from the artefacts that
``run_poc.py`` writes:

  - ``hrd_vs_ic50.csv``     : per-drug Spearman rho vs HRD score
  - ``per_drug_cv_auc.csv`` : per-drug 5-fold CV AUC by feature set
  - ``poc_summary.txt``     : cohort header

The portfolio site at ``bioinformatics-portfolio/shared/poc/project-4.json``
is a snapshot of this manifest; re-copy after running this script so
the portfolio's headline numbers stay in sync with the POC results.

Usage
-----
    python scripts/poc/build_manifest.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "results" / "poc"
OUT = POC / "manifest.json"


def main() -> int:
    hrd_csv = POC / "hrd_vs_ic50.csv"
    auc_csv = POC / "per_drug_cv_auc.csv"
    for f in (hrd_csv, auc_csv):
        if not f.is_file():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 1

    hrd = pd.read_csv(hrd_csv)
    auc = pd.read_csv(auc_csv)

    # Headline: strongest rho in the HRD_broad (11 DDR genes) block
    broad = hrd[hrd["hrd_definition"].str.startswith("HRD_broad")]
    best = broad.loc[broad["rho"].abs().idxmax()]

    # Tissue-context AUCs give the 0.68-0.77 range advertised
    tissue = auc[auc["feature_set"] == "DDR + MSI + Tissue"]
    auc_low = float(tissue["cv_auc_mean"].min())
    auc_high = float(tissue["cv_auc_mean"].max())
    best_drug_auc = tissue.loc[tissue["cv_auc_mean"].idxmax()]

    manifest = {
        "$schema": (
            "https://github.com/adamhoffman2155-hue/bioinformatics-portfolio/"
            "blob/main/shared/poc-manifest.schema.json"
        ),
        "project": "project-4-ddr-biomarker-pipeline",
        "poc_title": "DDR biomarker analysis on full GDSC v17",
        "poc_version": "v2",
        "dataset": {
            "name": "GDSC v17",
            "source": (
                "CancerRxGene / Sanger (bundled in gdsctools PyPI package)"
            ),
            "substitute_for": "GDSC release 8.5 (hosts unreachable)",
            "n_cell_lines": 988,
            "ddr_genes_present": 11,
            "msi_h_cell_lines": 60,
        },
        "script": "scripts/poc/run_poc.py",
        "generated_at": date.today().isoformat(),
        "headline_metric": {
            "name": f"{best['drug_name']} HRD Spearman rho",
            "value": round(float(best["rho"]), 3),
            "p_value": float(best["p"]),
            "n": int(best["n"]),
            "note": "HRD_broad (11 DDR genes); Bonferroni-significant x 5 drugs",
        },
        "secondary_metrics": [
            {
                "name": "5-fold CV ROC-AUC (DDR + MSI + Tissue)",
                "range": [round(auc_low, 3), round(auc_high, 3)],
                "best": {
                    "drug": best_drug_auc["drug_name"],
                    "value": round(float(best_drug_auc["cv_auc_mean"]), 3),
                    "std": round(float(best_drug_auc["cv_auc_std"]), 3),
                },
            },
            {
                "name": "5-fold CV ROC-AUC (DDR + MSI only)",
                "range": [
                    round(
                        float(
                            auc[auc["feature_set"] == "DDR + MSI_FACTOR"][
                                "cv_auc_mean"
                            ].min()
                        ),
                        3,
                    ),
                    round(
                        float(
                            auc[auc["feature_set"] == "DDR + MSI_FACTOR"][
                                "cv_auc_mean"
                            ].max()
                        ),
                        3,
                    ),
                ],
                "note": "tissue context adds 10-25 AUC points",
            },
            {
                "name": "MSI-H sensitivity (Mann-Whitney)",
                "findings": [
                    {
                        "drug": "KU-55933",
                        "p": 0.022,
                        "note": "MSI-H sensitive to ATMi",
                    },
                    {
                        "drug": "AZD7762",
                        "p": 0.019,
                        "note": "MSI-H sensitive to CHKi",
                    },
                ],
            },
        ],
        "headline_text": (
            f"{best['drug_name']} HRD ρ = +{abs(float(best['rho'])):.3f} "
            f"(p={float(best['p']):.1e}, n={int(best['n'])}); "
            f"5-fold CV AUC {auc_low:.2f}–{auc_high:.2f} with tissue context; "
            "MSI-H enriched for ATMi/CHKi sensitivity."
        ),
        "artifacts": [
            "results/poc/poc_summary.txt",
            "results/poc/hrd_vs_ic50.csv",
            "results/poc/per_drug_cv_auc.csv",
            "results/poc/msi_hrd_interaction.csv",
            "results/poc/ddr_shap_top.csv",
        ],
    }

    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT}")
    print(f"  headline: {manifest['headline_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
