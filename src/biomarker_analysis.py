"""
Biomarker analysis module for the DDR Biomarker Pipeline.

Provides BiomarkerAnalyzer with:
  - SHAP value computation (TreeExplainer / LinearExplainer)
  - Top biomarker identification by mean |SHAP|
  - Mann-Whitney U test for biomarker-drug response association
  - Cohen's d effect size computation
  - Seaborn clustermap of mutation status vs drug sensitivity
  - Comprehensive biomarker report generation

Typical usage::

    from src.biomarker_analysis import BiomarkerAnalyzer

    analyzer = BiomarkerAnalyzer(output_dir="results/olaparib/biomarkers")
    shap_values = analyzer.compute_shap_values(model, X_test)
    top_features = analyzer.identify_top_biomarkers(shap_values, X_test, top_n=20)
    assoc_df = analyzer.test_biomarker_association(X_test, y_ic50_continuous)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


class BiomarkerAnalyzer:
    """
    SHAP-based biomarker discovery and statistical association testing.

    Parameters
    ----------
    output_dir : str
    drug_name : str
    top_n : int
        Number of top biomarkers to report.
    dpi : int
    """

    def __init__(
        self,
        output_dir: str = "results/biomarkers",
        drug_name: str = "",
        top_n: int = 20,
        dpi: int = 150,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.drug_name = drug_name
        self.top_n = top_n
        self.dpi = dpi

        self._shap_values: Optional[np.ndarray] = None
        self._shap_feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # SHAP analysis
    # ------------------------------------------------------------------

    def compute_shap_values(
        self,
        model: Any,
        X: pd.DataFrame,
        max_background_samples: int = 100,
        use_tree_explainer: bool = True,
    ) -> np.ndarray:
        """
        Compute SHAP values for a fitted model.

        Automatically selects TreeExplainer for tree-based models
        (RandomForest, GradientBoosting) and LinearExplainer for
        linear models (LogisticRegression, ElasticNet/SGD).

        Parameters
        ----------
        model : sklearn estimator or Pipeline
        X : pd.DataFrame
            Feature matrix to explain.
        max_background_samples : int
            Number of background samples for kernel/linear explainer.
        use_tree_explainer : bool
            If True, try TreeExplainer first.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Run: pip install shap")

        self._shap_feature_names = list(X.columns)
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        # Unwrap Pipeline to get the actual estimator
        estimator = self._unwrap_pipeline(model)

        shap_values = None

        if use_tree_explainer and hasattr(estimator, "feature_importances_"):
            # Tree-based: RandomForest or GradientBoosting
            try:
                explainer = shap.TreeExplainer(estimator)
                raw = explainer.shap_values(X_arr)
                # For binary classifiers, shap_values returns [class0, class1]
                if isinstance(raw, list) and len(raw) == 2:
                    shap_values = raw[1]  # Class 1 (sensitive)
                else:
                    shap_values = raw
                logger.info("SHAP TreeExplainer: computed values for %d samples", len(X_arr))
            except Exception as e:
                logger.warning("TreeExplainer failed: %s. Falling back to LinearExplainer.", e)

        if shap_values is None and hasattr(estimator, "coef_"):
            # Linear model: LogisticRegression or SGDClassifier
            try:
                # Use a random background dataset
                n_bg = min(max_background_samples, X_arr.shape[0])
                rng = np.random.default_rng(42)
                bg_idx = rng.choice(X_arr.shape[0], n_bg, replace=False)
                background = X_arr[bg_idx]

                explainer = shap.LinearExplainer(estimator, background)
                shap_values = explainer.shap_values(X_arr)
                logger.info("SHAP LinearExplainer: computed values for %d samples", len(X_arr))
            except Exception as e:
                logger.warning("LinearExplainer failed: %s. Using KernelExplainer.", e)

        if shap_values is None:
            # Fallback: KernelExplainer (slow but universal)
            logger.info("Using KernelExplainer (slow for large datasets).")
            n_bg = min(50, X_arr.shape[0])
            rng = np.random.default_rng(42)
            bg_idx = rng.choice(X_arr.shape[0], n_bg, replace=False)
            background = shap.kmeans(X_arr[bg_idx], 10)

            def predict_proba_fn(x):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(x)[:, 1]
                return model.decision_function(x)

            explainer = shap.KernelExplainer(predict_proba_fn, background)
            shap_values = explainer.shap_values(X_arr, nsamples=100)

        self._shap_values = np.array(shap_values)
        logger.info("SHAP values shape: %s", self._shap_values.shape)
        return self._shap_values

    def identify_top_biomarkers(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rank features by mean absolute SHAP value.

        Parameters
        ----------
        shap_values : np.ndarray of shape (n_samples, n_features)
        X : pd.DataFrame
            Feature matrix (used for feature names).
        top_n : int, optional

        Returns
        -------
        pd.DataFrame with columns: feature, mean_abs_shap, std_abs_shap,
            mean_shap (signed), positive_fraction.
        """
        if top_n is None:
            top_n = self.top_n

        feature_names = list(X.columns)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        std_abs_shap = np.abs(shap_values).std(axis=0)
        mean_shap = shap_values.mean(axis=0)
        positive_fraction = (shap_values > 0).mean(axis=0)

        df = pd.DataFrame({
            "feature": feature_names[:len(mean_abs_shap)],
            "mean_abs_shap": mean_abs_shap,
            "std_abs_shap": std_abs_shap,
            "mean_shap": mean_shap,
            "positive_fraction": positive_fraction,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)

        df["rank"] = range(1, len(df) + 1)
        df = df.reset_index(drop=True)

        logger.info(
            "Top biomarker: %s (mean |SHAP| = %.4f)",
            df.iloc[0]["feature"] if len(df) > 0 else "N/A",
            df.iloc[0]["mean_abs_shap"] if len(df) > 0 else 0.0,
        )
        return df

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        top_n: Optional[int] = None,
        save: bool = True,
    ) -> plt.Figure:
        """
        Plot SHAP summary (beeswarm) using matplotlib.

        Parameters
        ----------
        shap_values : np.ndarray
        X : pd.DataFrame
        top_n : int, optional
        save : bool

        Returns
        -------
        matplotlib Figure
        """
        if top_n is None:
            top_n = self.top_n

        feature_names = list(X.columns)
        X_arr = X.values

        # Select top features by mean |SHAP|
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_n]
        top_features = [feature_names[i] for i in top_idx]
        top_shap = shap_values[:, top_idx]
        top_X = X_arr[:, top_idx]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))

        # Build beeswarm-style strip plot
        y_pos = np.arange(top_n)
        for j, (feat, sv, xv) in enumerate(zip(top_features, top_shap.T, top_X.T)):
            # Color by feature value
            vmin, vmax = xv.min(), xv.max()
            colors = plt.cm.RdYlBu_r(
                (xv - vmin) / (vmax - vmin + 1e-8)
            )
            # Add jitter in y
            jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(sv))
            ax.scatter(sv, top_n - 1 - j + jitter, c=colors, s=8, alpha=0.6, rasterized=True)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace("_lof_mutation", "").replace("_", " ")
                            for f in reversed(top_features)], fontsize=9)
        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set_xlabel("SHAP Value (impact on model output)", fontsize=11)
        ax.set_title(
            f"SHAP Summary — {self.drug_name or 'Drug'} Sensitivity",
            fontsize=13, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        # Colorbar legend
        sm = plt.cm.ScalarMappable(cmap="RdYlBu_r",
                                    norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Feature value\n(low → high)", fontsize=9)

        fig.tight_layout()

        if save:
            path = self.output_dir / "shap_summary.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("SHAP summary plot saved: %s", path)

        return fig

    # ------------------------------------------------------------------
    # Statistical association testing
    # ------------------------------------------------------------------

    def test_biomarker_association(
        self,
        X: pd.DataFrame,
        ic50_continuous: pd.Series,
        binary_features: Optional[List[str]] = None,
        fdr_method: str = "fdr_bh",
    ) -> pd.DataFrame:
        """
        Run Mann-Whitney U test for each binary biomarker vs continuous IC50.

        Tests the null hypothesis that the IC50 distribution is the same
        in mutant vs wildtype cell lines for each gene/feature.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Binary (0/1) columns used as grouping variables.
        ic50_continuous : pd.Series
            Continuous LN_IC50 values (lower = more sensitive).
        binary_features : list of str, optional
            Subset of X columns to test. Auto-detected if None.
        fdr_method : str
            Multiple testing correction method ('fdr_bh' = Benjamini-Hochberg).

        Returns
        -------
        pd.DataFrame with columns:
            feature, n_mutant, n_wildtype, median_ic50_mutant,
            median_ic50_wildtype, mann_whitney_u, p_value, q_value,
            cohen_d, direction
        """
        # Auto-detect binary (0/1) features
        if binary_features is None:
            binary_features = [
                c for c in X.columns
                if X[c].nunique() == 2 and set(X[c].unique()).issubset({0, 1})
            ]

        # Align IC50 with X
        shared_idx = X.index.intersection(ic50_continuous.index)
        X_aligned = X.loc[shared_idx]
        ic50_aligned = ic50_continuous.loc[shared_idx]

        rows = []
        for feat in binary_features:
            mutant_mask = X_aligned[feat] == 1
            wt_mask = X_aligned[feat] == 0

            ic50_mut = ic50_aligned[mutant_mask].dropna()
            ic50_wt = ic50_aligned[wt_mask].dropna()

            if len(ic50_mut) < 3 or len(ic50_wt) < 3:
                continue  # Skip features with too few samples in one group

            # Mann-Whitney U test
            try:
                stat, pval = mannwhitneyu(ic50_mut, ic50_wt, alternative="two-sided")
            except ValueError:
                continue

            # Cohen's d
            cohen_d = self._compute_cohens_d(ic50_mut.values, ic50_wt.values)

            # Direction of effect
            direction = "lower_in_mutant" if ic50_mut.median() < ic50_wt.median() else "higher_in_mutant"

            rows.append({
                "feature": feat,
                "n_mutant": len(ic50_mut),
                "n_wildtype": len(ic50_wt),
                "median_ic50_mutant": float(ic50_mut.median()),
                "median_ic50_wildtype": float(ic50_wt.median()),
                "delta_median": float(ic50_wt.median() - ic50_mut.median()),  # positive = mutant more sensitive
                "mann_whitney_u": float(stat),
                "p_value": float(pval),
                "cohen_d": float(cohen_d),
                "direction": direction,
            })

        if not rows:
            logger.warning("No association tests could be run.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("p_value")

        # FDR correction
        if len(df) > 0:
            _, q_values, _, _ = multipletests(df["p_value"], method=fdr_method)
            df["q_value"] = q_values
            df["significant"] = df["q_value"] < 0.05

        df = df.sort_values("p_value").reset_index(drop=True)
        n_sig = (df["q_value"] < 0.05).sum()
        logger.info(
            "Biomarker association: %d features tested, %d significant (q < 0.05)",
            len(df), n_sig,
        )
        return df

    def compute_effect_sizes(
        self,
        X: pd.DataFrame,
        ic50_continuous: pd.Series,
        features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute Cohen's d effect sizes for biomarker-drug response relationships.

        Parameters
        ----------
        X : pd.DataFrame
        ic50_continuous : pd.Series
        features : list of str, optional

        Returns
        -------
        pd.DataFrame with columns: feature, cohen_d, abs_cohen_d, interpretation
        """
        if features is None:
            features = [
                c for c in X.columns
                if X[c].nunique() == 2 and set(X[c].unique()).issubset({0, 1})
            ]

        shared_idx = X.index.intersection(ic50_continuous.index)
        X_aligned = X.loc[shared_idx]
        ic50_aligned = ic50_continuous.loc[shared_idx]

        rows = []
        for feat in features:
            mutant_mask = X_aligned[feat] == 1
            wt_mask = X_aligned[feat] == 0
            ic50_mut = ic50_aligned[mutant_mask].dropna().values
            ic50_wt = ic50_aligned[wt_mask].dropna().values

            if len(ic50_mut) < 2 or len(ic50_wt) < 2:
                continue

            d = self._compute_cohens_d(ic50_mut, ic50_wt)
            abs_d = abs(d)
            interpretation = self._interpret_cohen_d(abs_d)

            rows.append({
                "feature": feat,
                "cohen_d": float(d),
                "abs_cohen_d": float(abs_d),
                "interpretation": interpretation,
                "n_mutant": len(ic50_mut),
                "n_wildtype": len(ic50_wt),
            })

        if not rows:
            return pd.DataFrame()

        return (
            pd.DataFrame(rows)
            .sort_values("abs_cohen_d", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_biomarker_heatmap(
        self,
        X: pd.DataFrame,
        ic50_continuous: pd.Series,
        top_features: Optional[List[str]] = None,
        n_features: int = 20,
        save: bool = True,
    ) -> Optional[plt.Figure]:
        """
        Seaborn clustermap of mutation status vs drug sensitivity.

        Rows = cell lines, columns = top DDR biomarkers (binary mutation status).
        Cell lines are annotated with IC50 quartile (sensitive/resistant).

        Parameters
        ----------
        X : pd.DataFrame
            Binary mutation feature matrix.
        ic50_continuous : pd.Series
        top_features : list of str, optional
            Pre-selected feature names. Auto-selects by variance if None.
        n_features : int
            Number of features to display.
        save : bool

        Returns
        -------
        seaborn ClusterGrid (or None if too few samples)
        """
        # Select top features by variance (proxy for informativeness)
        shared_idx = X.index.intersection(ic50_continuous.index)
        X_aligned = X.loc[shared_idx].copy()
        ic50_aligned = ic50_continuous.loc[shared_idx]

        if len(X_aligned) < 5:
            logger.warning("Too few samples for heatmap (%d)", len(X_aligned))
            return None

        if top_features is None:
            binary_cols = [
                c for c in X_aligned.columns
                if X_aligned[c].nunique() == 2
                and set(X_aligned[c].unique()).issubset({0, 1})
            ]
            if not binary_cols:
                binary_cols = list(X_aligned.columns[:n_features])
            variances = X_aligned[binary_cols].var()
            top_features = variances.nlargest(n_features).index.tolist()

        # Subset and rename columns for display
        heat_data = X_aligned[top_features].copy()
        heat_data.columns = [
            c.replace("_lof_mutation", "").replace("_", " ")
            for c in heat_data.columns
        ]

        # IC50 quartile as row color
        q25 = ic50_aligned.quantile(0.25)
        q75 = ic50_aligned.quantile(0.75)

        def ic50_color(v):
            if v <= q25:
                return "#E53935"  # Red = sensitive
            elif v >= q75:
                return "#1E88E5"  # Blue = resistant
            else:
                return "#FDD835"  # Yellow = intermediate

        row_colors = ic50_aligned.map(ic50_color)

        try:
            g = sns.clustermap(
                heat_data,
                method="ward",
                metric="hamming",
                cmap="Greys",
                row_colors=row_colors,
                col_cluster=True,
                row_cluster=True,
                linewidths=0,
                figsize=(max(10, n_features * 0.6), min(20, len(heat_data) * 0.08 + 3)),
                cbar_kws={"label": "Mutation (1=LoF)"},
                xticklabels=True,
                yticklabels=False,
                vmin=0, vmax=1,
            )
            g.ax_heatmap.set_xlabel("DDR Gene (LoF Mutation)", fontsize=11)
            g.ax_heatmap.set_title(
                f"DDR Mutation Status vs {self.drug_name or 'Drug'} Sensitivity",
                fontsize=12, fontweight="bold", pad=20,
            )

            # Add legend for row colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#E53935", label="Sensitive (Q1 IC50)"),
                Patch(facecolor="#FDD835", label="Intermediate"),
                Patch(facecolor="#1E88E5", label="Resistant (Q3 IC50)"),
            ]
            g.ax_heatmap.legend(
                handles=legend_elements, loc="upper right",
                bbox_to_anchor=(1.15, 1.15), title="IC50 Quartile",
            )

            plt.tight_layout()

            if save:
                path = self.output_dir / "biomarker_heatmap.png"
                g.savefig(path, dpi=self.dpi, bbox_inches="tight")
                logger.info("Biomarker heatmap saved: %s", path)

            return g

        except Exception as e:
            logger.warning("Clustermap failed: %s", e)
            return None

    def plot_top_biomarker_boxplots(
        self,
        X: pd.DataFrame,
        ic50_continuous: pd.Series,
        top_features: Optional[List[str]] = None,
        n_features: int = 6,
        save: bool = True,
    ) -> plt.Figure:
        """
        Boxplots of IC50 stratified by mutation status for top biomarkers.

        Parameters
        ----------
        X : pd.DataFrame
        ic50_continuous : pd.Series
        top_features : list of str, optional
        n_features : int
        save : bool

        Returns
        -------
        matplotlib Figure
        """
        shared_idx = X.index.intersection(ic50_continuous.index)
        X_aligned = X.loc[shared_idx]
        ic50_aligned = ic50_continuous.loc[shared_idx]

        if top_features is None:
            binary_cols = [
                c for c in X_aligned.columns
                if X_aligned[c].nunique() == 2
            ]
            variances = X_aligned[binary_cols].var()
            top_features = variances.nlargest(n_features).index.tolist()

        top_features = top_features[:n_features]
        ncols = 3
        nrows = (len(top_features) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, feat in zip(axes, top_features):
            mutant_ic50 = ic50_aligned[X_aligned[feat] == 1].values
            wt_ic50 = ic50_aligned[X_aligned[feat] == 0].values

            # Filter non-empty
            groups = []
            labels = []
            if len(mutant_ic50) > 0:
                groups.append(mutant_ic50)
                labels.append(f"Mutant\n(n={len(mutant_ic50)})")
            if len(wt_ic50) > 0:
                groups.append(wt_ic50)
                labels.append(f"Wildtype\n(n={len(wt_ic50)})")

            bp = ax.boxplot(groups, patch_artist=True, notch=False, widths=0.5)
            colors = ["#E53935", "#90CAF9"]
            for patch, color in zip(bp["boxes"], colors[:len(groups)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("LN_IC50", fontsize=9)
            gene = feat.replace("_lof_mutation", "").replace("_", " ")
            ax.set_title(gene, fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            # Mann-Whitney p-value annotation
            if len(mutant_ic50) >= 3 and len(wt_ic50) >= 3:
                try:
                    _, pval = mannwhitneyu(mutant_ic50, wt_ic50, alternative="two-sided")
                    pval_str = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
                    ax.set_title(f"{gene}\n{pval_str}", fontsize=10, fontweight="bold")
                except Exception:
                    pass

        # Hide unused axes
        for ax in axes[len(top_features):]:
            ax.set_visible(False)

        fig.suptitle(
            f"IC50 by Mutation Status — {self.drug_name or 'Drug'}",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        if save:
            path = self.output_dir / "biomarker_boxplots.png"
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
            logger.info("Biomarker boxplots saved: %s", path)

        return fig

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_biomarker_report(
        self,
        top_biomarkers_df: pd.DataFrame,
        association_df: pd.DataFrame,
        effect_sizes_df: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> str:
        """
        Generate a text summary of biomarker analysis results.

        Parameters
        ----------
        top_biomarkers_df : pd.DataFrame
            From identify_top_biomarkers().
        association_df : pd.DataFrame
            From test_biomarker_association().
        effect_sizes_df : pd.DataFrame, optional
            From compute_effect_sizes().
        save : bool

        Returns
        -------
        str : Report text.
        """
        lines = [
            "=" * 70,
            f"BIOMARKER ANALYSIS REPORT",
            f"Drug: {self.drug_name or 'N/A'}",
            "=" * 70,
            "",
        ]

        # SHAP top features
        if len(top_biomarkers_df) > 0:
            lines += [
                "TOP BIOMARKERS BY SHAP IMPORTANCE",
                "-" * 40,
            ]
            for _, row in top_biomarkers_df.head(10).iterrows():
                gene = row["feature"].replace("_lof_mutation", "")
                direction = "sensitizing" if row["mean_shap"] < 0 else "resistant-associated"
                lines.append(
                    f"  {int(row['rank']):2d}. {gene:<15} "
                    f"mean|SHAP|={row['mean_abs_shap']:.4f}  ({direction})"
                )
            lines.append("")

        # Statistical associations
        if len(association_df) > 0:
            lines += [
                "STATISTICAL ASSOCIATIONS (Mann-Whitney U, BH-FDR)",
                "-" * 40,
            ]
            sig_df = association_df[association_df.get("q_value", association_df["p_value"]) < 0.05]
            if len(sig_df) == 0:
                lines.append("  No significant associations at q < 0.05.")
            else:
                for _, row in sig_df.head(10).iterrows():
                    gene = row["feature"].replace("_lof_mutation", "")
                    pval_str = f"p={row['p_value']:.2e}"
                    q_str = f"q={row.get('q_value', row['p_value']):.2e}"
                    lines.append(
                        f"  {gene:<15} {pval_str}  {q_str}  "
                        f"Cohen's d={row['cohen_d']:.2f}  "
                        f"n_mut={row['n_mutant']}"
                    )
            lines.append("")

        # Effect sizes
        if effect_sizes_df is not None and len(effect_sizes_df) > 0:
            lines += [
                "EFFECT SIZES (Cohen's d)",
                "-" * 40,
            ]
            for _, row in effect_sizes_df.head(10).iterrows():
                gene = row["feature"].replace("_lof_mutation", "")
                lines.append(
                    f"  {gene:<15} d={row['cohen_d']:+.3f}  "
                    f"({row['interpretation']})  n_mut={row['n_mutant']}"
                )
            lines.append("")

        lines.append("=" * 70)
        report = "\n".join(lines)
        print(report)

        if save:
            path = self.output_dir / "biomarker_report.txt"
            path.write_text(report)
            logger.info("Biomarker report saved: %s", path)

            # Save top biomarkers CSV
            if len(top_biomarkers_df) > 0:
                csv_path = self.output_dir / "top_biomarkers.csv"
                top_biomarkers_df.to_csv(csv_path, index=False)
                logger.info("Top biomarkers CSV saved: %s", csv_path)

            # Save association table CSV
            if len(association_df) > 0:
                assoc_path = self.output_dir / "biomarker_associations.csv"
                association_df.to_csv(assoc_path, index=False)
                logger.info("Association table saved: %s", assoc_path)

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size between two groups.

        d = (mean1 - mean2) / pooled_std
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0

        mean1, mean2 = group1.mean(), group2.mean()
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std < 1e-10:
            return 0.0

        return float((mean1 - mean2) / pooled_std)

    @staticmethod
    def _interpret_cohen_d(abs_d: float) -> str:
        """Return qualitative interpretation of Cohen's d."""
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        elif abs_d < 1.2:
            return "large"
        else:
            return "very large"

    @staticmethod
    def _unwrap_pipeline(model: Any) -> Any:
        """Extract base estimator from sklearn Pipeline."""
        if hasattr(model, "named_steps"):
            # Return the last step (classifier)
            steps = list(model.named_steps.values())
            return steps[-1]
        return model
