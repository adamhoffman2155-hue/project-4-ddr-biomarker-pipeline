"""
Feature engineering module for the DDR Biomarker Pipeline.

Transforms raw mutation, copy-number, and drug-response data into
an ML-ready feature matrix. Implements:

  - Binary mutation encoding (LoF per gene per cell line)
  - Mutation burden (total somatic load)
  - HRD composite score (HR pathway deficiency)
  - MSI status (mismatch repair deficiency)
  - DDR pathway activity scores (HR, NHEJ, MMR, FA sub-pathways)
  - IC50 normalization (z-score within tissue type)
  - Final feature matrix assembly

Typical usage::

    from src.feature_engineering import FeatureEngineer
    from config.config import PipelineConfig

    cfg = PipelineConfig()
    fe = FeatureEngineer(cfg)
    X, y = fe.prepare_feature_matrix(merged_df, drug_ic50_series)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDR sub-pathway gene sets
# ---------------------------------------------------------------------------

HR_GENES = [
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "BRIP1",
    "BARD1", "NBN", "MRE11", "RAD50", "RAD51", "RBBP8",
]

NHEJ_GENES = [
    "PRKDC",  # DNA-PKcs
    "LIG4", "XRCC4", "XRCC5", "XRCC6",
    "NHEJ1", "DCLRE1C",  # Artemis
]

MMR_GENES = ["MSH2", "MSH6", "MLH1", "PMS2", "MSH3"]

FA_GENES = [
    "FANCA", "FANCC", "FANCD1",  # FANCD1 = BRCA2
    "FANCD2", "FANCG", "FANCI", "FANCJ",  # FANCJ = BRIP1
    "FANCL", "FANCM", "FANCN",  # FANCN = PALB2
]

BER_GENES = ["PARP1", "PARP2", "XRCC1", "NEIL1", "NEIL2", "OGG1", "MUTYH"]

ATR_PATHWAY_GENES = ["ATR", "ATRIP", "CHEK1", "RPA1", "TopBP1"]
ATM_PATHWAY_GENES = ["ATM", "CHEK2", "H2AFX", "MDM2", "TP53"]

PATHWAY_GENE_SETS: Dict[str, List[str]] = {
    "hr_pathway": HR_GENES,
    "nhej_pathway": NHEJ_GENES,
    "mmr_pathway": MMR_GENES,
    "fa_pathway": FA_GENES,
    "ber_pathway": BER_GENES,
    "atr_pathway": ATR_PATHWAY_GENES,
    "atm_pathway": ATM_PATHWAY_GENES,
}


class FeatureEngineer:
    """
    Constructs feature matrices for DDR biomarker ML models.

    Parameters
    ----------
    config : PipelineConfig
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Core feature engineering methods
    # ------------------------------------------------------------------

    def encode_mutations_binary(
        self,
        mutations_df: pd.DataFrame,
        genes: Optional[List[str]] = None,
        lof_only: bool = True,
    ) -> pd.DataFrame:
        """
        Pivot long-format mutation data into a binary (cell line x gene) matrix.

        Each cell is 1 if the gene has a loss-of-function mutation in that
        cell line, 0 otherwise.

        Parameters
        ----------
        mutations_df : pd.DataFrame
            Long-format DepMap mutation table with columns:
            ModelID, HugoSymbol, VariantType.
        genes : list of str, optional
            Gene panel. Defaults to config DDR gene panel.
        lof_only : bool
            If True, keep only LoF variant classes.

        Returns
        -------
        pd.DataFrame
            Index = ModelID, columns = {gene}_lof_mutation, values = 0/1.
        """
        if genes is None:
            genes = self.config.ddr_genes

        df = mutations_df.copy()

        if lof_only and "VariantType" in df.columns:
            lof_classes = set(self.config.data.depmap_lof_classes)
            df = df[df["VariantType"].isin(lof_classes)]

        # Restrict to DDR genes
        df = df[df["HugoSymbol"].isin(genes)]

        # Pivot to binary matrix
        if len(df) == 0:
            logger.warning("No LoF mutations found for specified gene panel.")
            # Return empty matrix with correct columns
            return pd.DataFrame(
                columns=["ModelID"] + [f"{g}_lof_mutation" for g in genes]
            ).set_index("ModelID")

        binary_matrix = (
            df.groupby(["ModelID", "HugoSymbol"])
            .size()
            .gt(0)
            .astype(int)
            .unstack(fill_value=0)
        )
        binary_matrix.columns.name = None

        # Add missing genes as all-zero columns
        for gene in genes:
            if gene not in binary_matrix.columns:
                binary_matrix[gene] = 0

        binary_matrix = binary_matrix[genes]
        binary_matrix.columns = [f"{g}_lof_mutation" for g in binary_matrix.columns]

        logger.info(
            "Binary mutation matrix: %d cell lines x %d features",
            *binary_matrix.shape,
        )
        return binary_matrix

    def compute_mutation_burden(
        self,
        mutations_df: pd.DataFrame,
        log_transform: bool = True,
    ) -> pd.Series:
        """
        Compute total nonsynonymous somatic mutation count per cell line.

        Parameters
        ----------
        mutations_df : pd.DataFrame
            DepMap mutation table.
        log_transform : bool
            Apply log1p transformation to reduce skewness.

        Returns
        -------
        pd.Series
            Index = ModelID, values = mutation burden (possibly log-transformed).
        """
        # Count all mutations (or nonsynonymous if column available)
        nonsynonymous_classes = {
            "Missense_Mutation", "Nonsense_Mutation",
            "Frame_Shift_Del", "Frame_Shift_Ins",
            "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
        }

        df = mutations_df.copy()
        if "VariantType" in df.columns:
            df = df[df["VariantType"].isin(nonsynonymous_classes)]

        burden = df.groupby("ModelID").size().rename("mutation_burden")

        if log_transform:
            burden = np.log1p(burden).rename("mutation_burden_log")

        logger.info(
            "Mutation burden computed: %d cell lines, mean=%.2f, median=%.2f",
            len(burden), burden.mean(), burden.median(),
        )
        return burden

    def compute_hrd_score(
        self,
        binary_mutation_matrix: pd.DataFrame,
        hrd_genes: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Compute a composite Homologous Recombination Deficiency (HRD) score.

        The score is the count of HRD pathway genes with LoF mutations
        in each cell line (range 0 to len(hrd_genes)).

        A score >= 1 is considered HRD+.

        Parameters
        ----------
        binary_mutation_matrix : pd.DataFrame
            Output of encode_mutations_binary().
        hrd_genes : list of str, optional
            HRD pathway genes. Defaults to config.hrd_genes.

        Returns
        -------
        pd.Series
            Index = cell line (ModelID), values = HRD score (int).
        """
        if hrd_genes is None:
            hrd_genes = self.config.hrd_genes

        # Find matching columns in the matrix
        hrd_cols = [
            f"{g}_lof_mutation"
            for g in hrd_genes
            if f"{g}_lof_mutation" in binary_mutation_matrix.columns
        ]

        if not hrd_cols:
            logger.warning("No HRD gene columns found in mutation matrix.")
            return pd.Series(0, index=binary_mutation_matrix.index, name="hrd_score")

        hrd_score = binary_mutation_matrix[hrd_cols].sum(axis=1).rename("hrd_score")
        n_hrd_positive = (hrd_score >= 1).sum()
        logger.info(
            "HRD score: %d/%d cell lines are HRD+ (score >= 1)",
            n_hrd_positive, len(hrd_score),
        )
        return hrd_score

    def create_msi_status(
        self,
        binary_mutation_matrix: pd.DataFrame,
        mutations_df: Optional[pd.DataFrame] = None,
        mmr_genes: Optional[List[str]] = None,
        burden_threshold_percentile: float = 75.0,
    ) -> pd.Series:
        """
        Derive MSI (microsatellite instability) status.

        A cell line is classified MSI-H if it has:
          - At least one LoF mutation in an MMR gene (MSH2/MLH1/MSH6/PMS2), OR
          - Mutation burden above the 75th percentile (hypermutator phenotype).

        Parameters
        ----------
        binary_mutation_matrix : pd.DataFrame
        mutations_df : pd.DataFrame, optional
            Used to compute mutation burden if not already in the matrix.
        mmr_genes : list of str, optional
        burden_threshold_percentile : float
            Percentile cutoff for hypermutator classification.

        Returns
        -------
        pd.Series
            Index = cell line, values = 0/1 (MSS/MSI-H).
        """
        if mmr_genes is None:
            mmr_genes = self.config.mmr_genes

        mmr_cols = [
            f"{g}_lof_mutation"
            for g in mmr_genes
            if f"{g}_lof_mutation" in binary_mutation_matrix.columns
        ]

        # MSI from MMR gene mutation
        if mmr_cols:
            msi_from_mmr = binary_mutation_matrix[mmr_cols].any(axis=1).astype(int)
        else:
            msi_from_mmr = pd.Series(0, index=binary_mutation_matrix.index)

        # MSI from hypermutation burden
        msi_from_burden = pd.Series(0, index=binary_mutation_matrix.index)
        if mutations_df is not None:
            burden = self.compute_mutation_burden(mutations_df, log_transform=False)
            burden = burden.reindex(binary_mutation_matrix.index, fill_value=0)
            threshold = np.percentile(burden, burden_threshold_percentile)
            msi_from_burden = (burden >= threshold).astype(int)

        msi_status = (msi_from_mmr | msi_from_burden).rename("msi_status")
        n_msi_h = msi_status.sum()
        logger.info(
            "MSI-H: %d/%d cell lines (%.1f%%)",
            n_msi_h, len(msi_status), 100 * n_msi_h / max(len(msi_status), 1),
        )
        return msi_status

    def create_pathway_features(
        self,
        binary_mutation_matrix: pd.DataFrame,
        pathway_sets: Optional[Dict[str, List[str]]] = None,
    ) -> pd.DataFrame:
        """
        Compute DDR sub-pathway activity scores.

        Each score is the fraction of genes in the pathway with LoF mutations.

        Parameters
        ----------
        binary_mutation_matrix : pd.DataFrame
        pathway_sets : dict, optional
            Mapping of pathway name -> gene list.
            Defaults to PATHWAY_GENE_SETS defined in this module.

        Returns
        -------
        pd.DataFrame
            Index = cell line, columns = {pathway}_score.
        """
        if pathway_sets is None:
            pathway_sets = PATHWAY_GENE_SETS

        scores: Dict[str, pd.Series] = {}
        for pathway_name, genes in pathway_sets.items():
            cols = [
                f"{g}_lof_mutation"
                for g in genes
                if f"{g}_lof_mutation" in binary_mutation_matrix.columns
            ]
            if cols:
                # Fraction of pathway genes mutated (0 to 1)
                scores[f"{pathway_name}_score"] = (
                    binary_mutation_matrix[cols].sum(axis=1) / len(cols)
                )
            else:
                scores[f"{pathway_name}_score"] = pd.Series(
                    0.0, index=binary_mutation_matrix.index
                )

        pathway_df = pd.DataFrame(scores, index=binary_mutation_matrix.index)
        logger.info("Pathway features: %d scores computed", len(scores))
        return pathway_df

    def normalize_ic50(
        self,
        ic50_series: pd.Series,
        tissue_labels: Optional[pd.Series] = None,
        method: str = "zscore",
    ) -> pd.Series:
        """
        Normalize IC50 values.

        Parameters
        ----------
        ic50_series : pd.Series
            LN_IC50 values indexed by cell line name.
        tissue_labels : pd.Series, optional
            Tissue type per cell line (same index). If provided, z-score
            is computed within each tissue group.
        method : str
            'zscore' (default) or 'minmax'.

        Returns
        -------
        pd.Series
            Normalized IC50 values.
        """
        ic50 = ic50_series.copy().rename("ic50_normalized")

        if method == "zscore":
            if tissue_labels is not None and len(tissue_labels) > 0:
                # Z-score within tissue type
                combined = pd.DataFrame({
                    "ic50": ic50, "tissue": tissue_labels
                })
                normalized = combined.groupby("tissue")["ic50"].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
                ic50 = normalized.rename("ic50_normalized")
                logger.info("IC50 z-scored within %d tissue groups",
                            tissue_labels.nunique())
            else:
                mean, std = ic50.mean(), ic50.std()
                ic50 = ((ic50 - mean) / (std + 1e-8)).rename("ic50_normalized")
        elif method == "minmax":
            lo, hi = ic50.min(), ic50.max()
            ic50 = ((ic50 - lo) / (hi - lo + 1e-8)).rename("ic50_normalized")
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return ic50

    def binarize_ic50(
        self,
        ic50_series: pd.Series,
        quantile: float = 0.5,
        tissue_labels: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Convert continuous IC50 to binary sensitivity label.

        A cell line is labelled "sensitive" (1) if its IC50 is below the
        drug-specific (or tissue-specific) median.

        Parameters
        ----------
        ic50_series : pd.Series
        quantile : float
            Quantile cutoff (default: 0.5 = median).
        tissue_labels : pd.Series, optional
            If provided, threshold is computed per tissue.

        Returns
        -------
        pd.Series
            Binary series (1 = sensitive, 0 = resistant).
        """
        if tissue_labels is not None:
            combined = pd.DataFrame({"ic50": ic50_series, "tissue": tissue_labels})
            threshold = combined.groupby("tissue")["ic50"].transform(
                lambda x: x.quantile(quantile)
            )
            labels = (ic50_series <= threshold).astype(int).rename("sensitive")
        else:
            threshold = ic50_series.quantile(quantile)
            labels = (ic50_series <= threshold).astype(int).rename("sensitive")

        n_sensitive = labels.sum()
        logger.info(
            "Binarized IC50 (quantile=%.2f): %d sensitive / %d resistant",
            quantile, n_sensitive, len(labels) - n_sensitive,
        )
        return labels

    def prepare_feature_matrix(
        self,
        merged_df: pd.DataFrame,
        drug_ic50: pd.Series,
        include_pathway_scores: bool = True,
        include_mutation_burden: bool = True,
        include_hrd_score: bool = True,
        include_msi_status: bool = True,
        sensitivity_quantile: float = 0.5,
        fit_scaler: bool = True,
        scale_features: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Assemble the full ML feature matrix and binary sensitivity labels.

        Orchestrates all feature engineering steps:
        1. Extract binary mutation features from merged_df.
        2. Compute HRD score.
        3. Compute MSI status.
        4. Compute pathway activity scores.
        5. Add mutation burden.
        6. Binarize IC50 labels.
        7. Optionally scale features.

        Parameters
        ----------
        merged_df : pd.DataFrame
            Output of GDSCDataLoader.merge_datasets(); must contain
            CELL_LINE_NAME, LN_IC50, and {gene}_lof_mutation columns.
        drug_ic50 : pd.Series
            IC50 indexed by cell line name.
        include_pathway_scores : bool
        include_mutation_burden : bool
        include_hrd_score : bool
        include_msi_status : bool
        sensitivity_quantile : float
            Quantile for binarizing IC50 into sensitive/resistant.
        fit_scaler : bool
            Fit a new StandardScaler (training mode).
            Set False at inference time (uses already-fitted scaler).
        scale_features : bool
            Whether to apply StandardScaler to the feature matrix.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (X, y) where X is the feature matrix and y is 0/1 sensitivity.
        """
        df = merged_df.set_index("CELL_LINE_NAME") if "CELL_LINE_NAME" in merged_df.columns else merged_df.copy()

        # --- Extract binary mutation features ---
        mut_cols = [c for c in df.columns if c.endswith("_lof_mutation")]
        if not mut_cols:
            raise ValueError(
                "No '_lof_mutation' columns found in merged_df. "
                "Run GDSCDataLoader.merge_datasets() first."
            )
        X = df[mut_cols].copy()

        # --- HRD score ---
        if include_hrd_score:
            hrd = self.compute_hrd_score(X)
            X = X.join(hrd)

        # --- MSI status ---
        if include_msi_status:
            msi = self.create_msi_status(X)
            X = X.join(msi)

        # --- Pathway scores ---
        if include_pathway_scores:
            pathway_features = self.create_pathway_features(X)
            X = X.join(pathway_features)

        # --- Mutation burden (if available in merged_df) ---
        if include_mutation_burden and "mutation_burden_log" in df.columns:
            X = X.join(df[["mutation_burden_log"]])
        elif include_mutation_burden:
            logger.debug("mutation_burden_log not in merged_df; skipping.")

        # --- Tissue labels for stratification ---
        tissue_labels: Optional[pd.Series] = None
        if "TCGA_DESC" in df.columns:
            tissue_labels = df["TCGA_DESC"]

        # --- Binarize IC50 ---
        ic50_aligned = drug_ic50.reindex(X.index)
        missing_ic50 = ic50_aligned.isna().sum()
        if missing_ic50 > 0:
            logger.warning(
                "Dropping %d cell lines with missing IC50", missing_ic50
            )
            X = X[~ic50_aligned.isna()]
            ic50_aligned = ic50_aligned.dropna()
            if tissue_labels is not None:
                tissue_labels = tissue_labels.reindex(X.index)

        y = self.binarize_ic50(ic50_aligned, quantile=sensitivity_quantile,
                               tissue_labels=tissue_labels)

        # --- Align indices ---
        X, y = X.align(y, join="inner", axis=0)

        # --- Drop columns that are all-zero (no variation) ---
        zero_cols = X.columns[(X == 0).all()]
        if len(zero_cols) > 0:
            logger.debug("Dropping %d all-zero feature columns", len(zero_cols))
            X = X.drop(columns=zero_cols)

        # --- Fill any NaNs ---
        X = X.fillna(0)

        # --- Scale ---
        if scale_features:
            feature_names = X.columns.tolist()
            X_arr = X.values.astype(float)
            if fit_scaler:
                self._scaler = StandardScaler()
                X_arr = self._scaler.fit_transform(X_arr)
            elif self._scaler is not None:
                X_arr = self._scaler.transform(X_arr)
            else:
                logger.warning("No fitted scaler; skipping scaling.")
            X = pd.DataFrame(X_arr, index=X.index, columns=feature_names)

        logger.info(
            "Feature matrix ready: %d samples x %d features, "
            "%d sensitive / %d resistant",
            X.shape[0], X.shape[1], y.sum(), (y == 0).sum(),
        )
        return X, y

    def get_feature_names(self) -> List[str]:
        """Return feature names from the last call to prepare_feature_matrix()."""
        if self._scaler is not None and hasattr(self._scaler, "feature_names_in_"):
            return list(self._scaler.feature_names_in_)
        return []
