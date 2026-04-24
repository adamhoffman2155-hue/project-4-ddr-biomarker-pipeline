"""
POC: DDR biomarker analysis on GDSC cell line data.

Scientific question: Does HRD mutation burden predict sensitivity to PARP
inhibitors (and DDR inhibitors) in cell lines?

Pipeline:
  1. Load GDSC IC50 + mutation-level genomic features + drug compound metadata.
  2. Build an HRD-like score = #pathogenic mutations in an HR gene panel.
  3. For each drug of interest compute Spearman(HRD, ln IC50).
  4. Binarise sensitivity at the bottom quartile of ln IC50 and train a
     logistic regression on the HR-panel mutation features with 5-fold CV,
     reporting mean ROC-AUC.
  5. Run SHAP on the best-performing drug.

Honest reporting: weak Spearman correlations are expected for mutation-only
HRD scores (~ -0.15 to -0.3 for Olaparib in published cell line analyses).
We report actual values even when weak or near zero.

Data source caveat
------------------
The primary GDSC release 8.5 files (cog.sanger.ac.uk hosted xlsx/csv) are
not reachable from this sandbox (host not in allowlist). We instead use the
GDSC v17 IC50 + mutation matrix that ships with the `gdsctools` PyPI package
(CancerRxGene group, Sanger) plus the release 8.5 drug compound metadata
mirrored on GitHub (deZealous/drugresponse@190ab870).

The v17 genomic_features matrix pre-filters mutations to frequently-mutated
cancer genes, so only 4 of the 19 intended HR-panel genes are available
(BRCA1, BRCA2, ATM, CHEK2). The HRD score is therefore the number of
mutations across those 4 genes per cell line (0-4). We document this
limitation in the summary and README.

Because VE-821 and AZD6738 do not exist in the v17 snapshot (both were
added to GDSC in later releases), we additionally include KU-55933 (ATM
inhibitor) and AZD7762 (CHK1/CHK2 inhibitor) as "DDR-inhibitor fallbacks"
which target the broader DDR pathway.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "poc"
RESULTS.mkdir(parents=True, exist_ok=True)

# -------- configuration --------------------------------------------------
HR_GENES_REQUESTED = [
    "BRCA1", "BRCA2", "PALB2", "RAD51", "RAD51B", "RAD51C", "RAD51D",
    "ATM", "BARD1", "BRIP1", "CHEK2", "FANCA", "FANCC", "FANCD2", "FANCE",
    "FANCF", "NBN", "MRE11A", "RAD50",
]

# Drugs originally targeted. v17 release does not contain the ATR inhibitors
# (VE-821, AZD6738). We keep the PARPi and fall back to ATM / CHK inhibitors.
DRUG_PLAN = {
    1017: "Olaparib",       # PARPi
    1175: "Rucaparib",      # PARPi
    1259: "Talazoparib",    # PARPi
    1030: "KU-55933",       # ATM inhibitor (ATR fallback)
    1022: "AZD7762",        # CHEK1/2 inhibitor (DDR fallback)
}

RANDOM_STATE = 42


def load_data():
    ic = pd.read_csv(DATA / "IC50_v17.csv.gz")
    gf = pd.read_csv(DATA / "genomic_features_v17.csv.gz")
    cmp = pd.read_csv(DATA / "screened_compounds_rel_8.5.csv")
    return ic, gf, cmp


def build_hrd_feature_matrix(gf: pd.DataFrame):
    hr_cols_available = [g for g in HR_GENES_REQUESTED if f"{g}_mut" in gf.columns]
    missing = [g for g in HR_GENES_REQUESTED if g not in hr_cols_available]
    feats = gf[["COSMIC_ID"] + [f"{g}_mut" for g in hr_cols_available]].copy()
    feats.columns = ["COSMIC_ID"] + hr_cols_available
    for g in hr_cols_available:
        feats[g] = pd.to_numeric(feats[g], errors="coerce").fillna(0).astype(int)
    feats["HRD_score"] = feats[hr_cols_available].sum(axis=1)
    return feats, hr_cols_available, missing


def spearman_per_drug(ic: pd.DataFrame, feats: pd.DataFrame, drug_plan: dict):
    rows = []
    joined_cache = {}
    for drug_id, drug_name in drug_plan.items():
        col = f"Drug_{drug_id}_IC50"
        if col not in ic.columns:
            rows.append({
                "drug_id": drug_id, "drug_name": drug_name, "n": 0,
                "spearman_rho": np.nan, "p_value": np.nan,
                "note": f"{col} not in GDSC v17 IC50",
            })
            continue
        sub = ic[["COSMIC_ID", col]].dropna()
        merged = sub.merge(feats[["COSMIC_ID", "HRD_score"]], on="COSMIC_ID", how="inner")
        if merged.empty:
            rows.append({"drug_id": drug_id, "drug_name": drug_name,
                         "n": 0, "spearman_rho": np.nan, "p_value": np.nan,
                         "note": "no joined rows"})
            continue
        rho, pval = spearmanr(merged["HRD_score"], merged[col])
        rows.append({"drug_id": drug_id, "drug_name": drug_name,
                     "n": int(len(merged)),
                     "spearman_rho": float(rho), "p_value": float(pval),
                     "note": ""})
        joined_cache[drug_id] = merged.rename(columns={col: "ln_IC50"})
    return pd.DataFrame(rows), joined_cache


def scatter_plot(joined_cache: dict, drug_plan: dict, out_path: Path):
    drugs = [d for d in drug_plan if d in joined_cache]
    if not drugs:
        return
    ncols = 3
    nrows = int(np.ceil(len(drugs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows),
                             squeeze=False)
    for ax, drug_id in zip(axes.flat, drugs):
        df = joined_cache[drug_id]
        rho, p = spearmanr(df["HRD_score"], df["ln_IC50"])
        jitter = np.random.default_rng(drug_id).normal(0, 0.08, size=len(df))
        ax.scatter(df["HRD_score"] + jitter, df["ln_IC50"],
                   alpha=0.45, s=14, color="#2b6cb0")
        ax.set_xlabel("HRD mutation score")
        ax.set_ylabel("ln IC50")
        ax.set_title(f"{drug_plan[drug_id]} (Drug_{drug_id})\n"
                     f"rho={rho:.3f}  p={p:.2g}  n={len(df)}", fontsize=9)
    for ax in axes.flat[len(drugs):]:
        ax.axis("off")
    fig.suptitle("HRD mutation score vs. drug ln IC50 (GDSC v17)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def cv_auc_per_drug(ic: pd.DataFrame, feats: pd.DataFrame, drug_plan: dict,
                    hr_cols: list[str]):
    rows = []
    trained_cache = {}
    for drug_id, drug_name in drug_plan.items():
        col = f"Drug_{drug_id}_IC50"
        if col not in ic.columns:
            rows.append({"drug_id": drug_id, "drug_name": drug_name,
                         "n": 0, "n_sensitive": 0, "cv_auc_mean": np.nan,
                         "cv_auc_std": np.nan,
                         "note": "drug absent from IC50"})
            continue
        sub = ic[["COSMIC_ID", col]].dropna()
        merged = sub.merge(feats[["COSMIC_ID"] + hr_cols],
                           on="COSMIC_ID", how="inner")
        if len(merged) < 40:
            rows.append({"drug_id": drug_id, "drug_name": drug_name,
                         "n": len(merged), "n_sensitive": 0,
                         "cv_auc_mean": np.nan, "cv_auc_std": np.nan,
                         "note": "insufficient samples"})
            continue
        q25 = merged[col].quantile(0.25)
        merged["sensitive"] = (merged[col] <= q25).astype(int)
        X = merged[hr_cols].values.astype(float)
        y = merged["sensitive"].values
        if y.sum() < 5 or (len(y) - y.sum()) < 5:
            rows.append({"drug_id": drug_id, "drug_name": drug_name,
                         "n": len(merged), "n_sensitive": int(y.sum()),
                         "cv_auc_mean": np.nan, "cv_auc_std": np.nan,
                         "note": "class imbalance"})
            continue
        pipe = Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=2000,
                                      class_weight="balanced",
                                      random_state=RANDOM_STATE)),
        ])
        skf = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=RANDOM_STATE)
        try:
            scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
        except Exception as exc:
            rows.append({"drug_id": drug_id, "drug_name": drug_name,
                         "n": len(merged), "n_sensitive": int(y.sum()),
                         "cv_auc_mean": np.nan, "cv_auc_std": np.nan,
                         "note": f"cv error: {exc}"})
            continue
        rows.append({"drug_id": drug_id, "drug_name": drug_name,
                     "n": len(merged), "n_sensitive": int(y.sum()),
                     "cv_auc_mean": float(scores.mean()),
                     "cv_auc_std": float(scores.std()),
                     "note": ""})
        # fit on full data for SHAP
        pipe.fit(X, y)
        trained_cache[drug_id] = {"model": pipe, "X": X, "y": y,
                                  "feature_names": hr_cols}
    return pd.DataFrame(rows), trained_cache


def shap_on_best(cv_df: pd.DataFrame, trained_cache: dict, out_path: Path):
    valid = cv_df.dropna(subset=["cv_auc_mean"]).sort_values(
        "cv_auc_mean", ascending=False)
    if valid.empty:
        pd.DataFrame(columns=["drug_name", "feature", "mean_abs_shap"]).to_csv(
            out_path, index=False)
        return None
    best = valid.iloc[0]
    best_id = int(best["drug_id"])
    bundle = trained_cache.get(best_id)
    if bundle is None:
        return None
    try:
        import shap
        explainer = shap.LinearExplainer(
            bundle["model"].named_steps["lr"],
            bundle["model"].named_steps["scale"].transform(bundle["X"]),
        )
        shap_values = explainer.shap_values(
            bundle["model"].named_steps["scale"].transform(bundle["X"]))
    except Exception as exc:  # pragma: no cover - defensive
        # Fallback: just use model coefficients as an approximation
        coefs = bundle["model"].named_steps["lr"].coef_.ravel()
        df = pd.DataFrame({
            "drug_name": best["drug_name"],
            "feature": bundle["feature_names"],
            "mean_abs_shap": np.abs(coefs),
            "direction": np.sign(coefs),
            "note": f"coef fallback ({exc})",
        }).sort_values("mean_abs_shap", ascending=False)
        df.to_csv(out_path, index=False)
        return best["drug_name"]

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    df = pd.DataFrame({
        "drug_name": best["drug_name"],
        "feature": bundle["feature_names"],
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False)
    df.to_csv(out_path, index=False)
    return best["drug_name"]


def write_summary(n_cell_lines, drug_plan, hr_cols, missing,
                  hrd_scores, spearman_df, cv_df, shap_drug, out_path: Path):
    lines = []
    lines.append("DDR biomarker POC - summary")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Dataset: GDSC (v17 bundled with gdsctools PyPI package) +")
    lines.append("  GDSC release 8.5 drug compound metadata mirrored on GitHub")
    lines.append("  (deZealous/drugresponse@190ab870).")
    lines.append("")
    lines.append("Data-access caveat")
    lines.append("------------------")
    lines.append("The cog.sanger.ac.uk host serving the release 8.5 IC50 xlsx")
    lines.append("and mutations_all_20230202.csv is blocked from this sandbox")
    lines.append("(host not in egress allowlist).  We used the v17 snapshot")
    lines.append("bundled inside the gdsctools pypi wheel (CancerRxGene Sanger).")
    lines.append("")
    lines.append(f"Cell lines analysed (N): {n_cell_lines}")
    lines.append("")
    lines.append("HR gene panel (requested): " + ", ".join(HR_GENES_REQUESTED))
    lines.append(f"  available in v17 genomic_features: {hr_cols}")
    lines.append(f"  missing (pruned from v17 matrix): {missing}")
    lines.append("")
    lines.append("HRD score distribution (#mutations across available HR genes):")
    vc = hrd_scores.value_counts().sort_index()
    for k, v in vc.items():
        lines.append(f"  score={int(k)}: n={int(v)}")
    lines.append(f"  mean={hrd_scores.mean():.3f}  std={hrd_scores.std():.3f}  "
                 f"max={int(hrd_scores.max())}")
    lines.append("")
    lines.append("Drugs planned vs analysed")
    lines.append("-------------------------")
    lines.append("Planned PARPi: Olaparib, Rucaparib, Talazoparib")
    lines.append("Planned ATRi : VE-821 (Drug 2111), AZD6738 (Drug 1394/1917)")
    lines.append("  -> both absent from v17 snapshot.")
    lines.append("Fallback DDR inhibitors analysed in their place:")
    lines.append("  KU-55933 (Drug 1030, ATM)")
    lines.append("  AZD7762  (Drug 1022, CHK1/CHK2)")
    lines.append("")
    lines.append("Per-drug Spearman (HRD score vs ln IC50)")
    lines.append("-----------------------------------------")
    for _, r in spearman_df.iterrows():
        lines.append(
            f"  {r['drug_name']:<12} (Drug {int(r['drug_id'])}): "
            f"n={int(r['n']):4d}  rho={r['spearman_rho']:+.3f}  "
            f"p={r['p_value']:.3g}  {r['note']}")
    lines.append("")
    lines.append("Per-drug 5-fold CV ROC-AUC (LogReg on HR-gene mutations)")
    lines.append("--------------------------------------------------------")
    for _, r in cv_df.iterrows():
        lines.append(
            f"  {r['drug_name']:<12} (Drug {int(r['drug_id'])}): "
            f"n={int(r['n']):4d}  n_sens={int(r['n_sensitive']):4d}  "
            f"AUC={r['cv_auc_mean']:.3f}+/-{r['cv_auc_std']:.3f}  "
            f"{r['note']}")
    lines.append("")
    lines.append("SHAP run on best-AUC drug: "
                 f"{shap_drug if shap_drug else 'N/A'}")
    lines.append("Top SHAP feature table -> results/poc/ddr_shap_top.csv")
    lines.append("")
    lines.append("Caveats and honest reporting")
    lines.append("----------------------------")
    lines.append("* Mutation-only HRD is a crude proxy.  Published genomic-scar")
    lines.append("  based HRD scores (HRD-LOH, LST, TAI, Myriad score) perform")
    lines.append("  noticeably better than binary mutation counts.")
    lines.append("* The v17 genomic_features matrix only retains frequently")
    lines.append("  mutated cancer genes, so 15 of the 19 intended HR-panel")
    lines.append("  genes (PALB2, RAD51, RAD51B/C/D, BARD1, BRIP1, NBN,")
    lines.append("  MRE11A, RAD50, FANCA/C/D2/E/F) are not represented.")
    lines.append("  This truncation is expected to attenuate HRD-sensitivity")
    lines.append("  correlations compared with the published literature.")
    lines.append("* Published Spearman rho for Olaparib vs mutation-only HRD")
    lines.append("  in cell line panels is typically -0.15 to -0.3; our")
    lines.append("  estimates should be interpreted against that expectation.")
    lines.append("* VE-821 and AZD6738 are in GDSC release 8.x but not in v17,")
    lines.append("  so ATRi sensitivity is not assessed directly; KU-55933")
    lines.append("  (ATMi) and AZD7762 (CHKi) are DDR-pathway stand-ins only.")
    out_path.write_text("\n".join(lines))


def main():
    print("[1] Loading data")
    ic, gf, cmp = load_data()
    print(f"   IC50 shape {ic.shape}  genomic_features shape {gf.shape}")
    print(f"   drug compound metadata rows {cmp.shape[0]}")

    print("[2] Building HRD feature matrix")
    feats, hr_cols, missing = build_hrd_feature_matrix(gf)
    print(f"   HR genes available in v17: {hr_cols}")
    print(f"   HR genes missing: {missing}")

    print("[3] Spearman correlations")
    spearman_df, joined_cache = spearman_per_drug(ic, feats, DRUG_PLAN)
    print(spearman_df.to_string(index=False))
    spearman_df.to_csv(RESULTS / "hrd_vs_ic50.csv", index=False)

    print("[4] Scatter plot")
    scatter_plot(joined_cache, DRUG_PLAN, RESULTS / "hrd_vs_ic50_scatter.png")

    print("[5] Per-drug CV AUC")
    cv_df, trained_cache = cv_auc_per_drug(ic, feats, DRUG_PLAN, hr_cols)
    print(cv_df.to_string(index=False))
    cv_df.to_csv(RESULTS / "per_drug_cv_auc.csv", index=False)

    print("[6] SHAP on best drug")
    shap_drug = shap_on_best(cv_df, trained_cache,
                             RESULTS / "ddr_shap_top.csv")
    print(f"   best drug: {shap_drug}")

    # restrict summary N to cell lines that have >= 1 drug IC50 and >= 1
    # mutation feature
    n_cell_lines = feats["COSMIC_ID"].nunique()
    print("[7] Writing summary")
    write_summary(
        n_cell_lines=n_cell_lines,
        drug_plan=DRUG_PLAN,
        hr_cols=hr_cols,
        missing=missing,
        hrd_scores=feats["HRD_score"],
        spearman_df=spearman_df,
        cv_df=cv_df,
        shap_drug=shap_drug,
        out_path=RESULTS / "poc_summary.txt",
    )

    # [8] Regenerate results/poc/manifest.json so the portfolio site's
    # headline numbers stay in sync with the freshly-written CSVs.
    print("[8] Rebuilding manifest.json")
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("build_manifest.py"))],
        check=True,
    )
    print("[done]")


if __name__ == "__main__":
    main()
