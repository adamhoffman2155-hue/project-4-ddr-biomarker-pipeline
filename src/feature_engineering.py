"""
Feature engineering for the DDR Biomarker Pipeline.

Transforms raw mutation and IC50 data into a modelling-ready feature matrix
with biologically motivated composite features (HRD score, MSI status,
DDR burden).
"""

import pandas as pd

from .utils import setup_logging

logger = setup_logging(__name__)


def compute_hrd_score(
    mutation_row: pd.Series,
    hr_genes: list[str] | None = None,
) -> int:
    """Count pathogenic mutations in HR-pathway genes.

    A simple proxy for Homologous Recombination Deficiency: the more HR
    genes are mutated, the higher the score.

    Args:
        mutation_row: A single cell line's binary mutation profile.
        hr_genes: Gene symbols to consider.  Defaults to the standard panel.

    Returns:
        Integer count of mutated HR genes (0 .. len(hr_genes)).
    """
    if hr_genes is None:
        hr_genes = ["BRCA1", "BRCA2", "PALB2", "RAD51", "ATM", "ATR"]
    present = [g for g in hr_genes if g in mutation_row.index]
    return int(mutation_row[present].sum())


def compute_msi_status(
    mutation_row: pd.Series,
    mmr_genes: list[str] | None = None,
) -> int:
    """Determine microsatellite-instability status from MMR gene mutations.

    Args:
        mutation_row: A single cell line's binary mutation profile.
        mmr_genes: MMR gene symbols.  Defaults to MLH1, MSH2, MSH6.

    Returns:
        1 if any MMR gene is mutated, else 0.
    """
    if mmr_genes is None:
        mmr_genes = ["MLH1", "MSH2", "MSH6"]
    present = [g for g in mmr_genes if g in mutation_row.index]
    return int(mutation_row[present].any())


def compute_ddr_burden(
    mutation_row: pd.Series,
    ddr_genes: list[str] | None = None,
) -> int:
    """Total number of DDR-gene mutations for a cell line.

    Args:
        mutation_row: A single cell line's binary mutation profile.
        ddr_genes: Full DDR gene panel.

    Returns:
        Integer mutation count across the DDR panel.
    """
    if ddr_genes is None:
        ddr_genes = [
            "BRCA1",
            "BRCA2",
            "ATM",
            "ATR",
            "PALB2",
            "RAD51",
            "MLH1",
            "MSH2",
            "MSH6",
            "POLE",
            "ARID1A",
            "CDK12",
            "CHEK2",
        ]
    present = [g for g in ddr_genes if g in mutation_row.index]
    return int(mutation_row[present].sum())


def build_feature_matrix(
    mutation_df: pd.DataFrame,
    ic50_df: pd.DataFrame,
    drug_name: str,
    config: object,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the modelling feature matrix **X** and label vector **y**.

    Features include:
    * Individual binary mutation flags for each DDR gene
    * HRD score (composite)
    * MSI status (binary)
    * DDR burden (composite)

    The label is binary: 1 = sensitive (IC50 < threshold), 0 = resistant.

    Args:
        mutation_df: Binary mutation matrix (cell lines x genes).
        ic50_df: IC50 matrix (cell lines x drugs).
        drug_name: Column name in *ic50_df* to use as the response variable.
        config: A :class:`~config.config.PipelineConfig` instance.

    Returns:
        Tuple ``(X, y)`` where *X* is a DataFrame of features and *y* is a
        binary Series with the same index.

    Raises:
        ValueError: If *drug_name* is not found in *ic50_df*.
    """
    if drug_name not in ic50_df.columns:
        raise ValueError(
            f"Drug '{drug_name}' not found in IC50 data. Available: {list(ic50_df.columns)}"
        )

    # Align indices
    common = mutation_df.index.intersection(ic50_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping cell lines between mutation and IC50 data")

    mut = mutation_df.loc[common]
    ic50 = ic50_df.loc[common, drug_name]

    logger.info("Building feature matrix for %s: %d cell lines", drug_name, len(common))

    # --- Individual gene features ---
    ddr_genes = config.DDR_GENES
    gene_features = mut[[g for g in ddr_genes if g in mut.columns]].copy()

    # --- Composite features ---
    hrd_scores = mut.apply(lambda row: compute_hrd_score(row, config.HR_GENES), axis=1)
    msi_status = mut.apply(lambda row: compute_msi_status(row, config.MMR_GENES), axis=1)
    ddr_burden = mut.apply(lambda row: compute_ddr_burden(row, config.DDR_GENES), axis=1)

    X = gene_features.copy()
    X["hrd_score"] = hrd_scores
    X["msi_status"] = msi_status
    X["ddr_burden"] = ddr_burden

    # --- Label vector ---
    y = (ic50 < config.IC50_THRESHOLD).astype(int)
    y.name = f"{drug_name}_sensitive"

    # Log class balance
    n_sens = int(y.sum())
    n_res = len(y) - n_sens
    logger.info(
        "  Class balance \u2014 sensitive: %d (%.1f%%), resistant: %d (%.1f%%)",
        n_sens,
        100 * n_sens / len(y),
        n_res,
        100 * n_res / len(y),
    )

    return X, y


def add_interaction_features(
    X: pd.DataFrame,
    gene_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Add pairwise interaction terms for selected gene pairs.

    Args:
        X: Feature matrix (must contain the individual gene columns).
        gene_pairs: List of (gene_a, gene_b) tuples.  Defaults to
            biologically motivated pairs.

    Returns:
        Augmented DataFrame with new ``geneA_x_geneB`` columns.
    """
    if gene_pairs is None:
        gene_pairs = [
            ("BRCA1", "BRCA2"),
            ("ATM", "ATR"),
            ("MLH1", "MSH2"),
            ("BRCA1", "PALB2"),
        ]

    X_aug = X.copy()
    for g1, g2 in gene_pairs:
        if g1 in X.columns and g2 in X.columns:
            col_name = f"{g1}_x_{g2}"
            X_aug[col_name] = X[g1] * X[g2]
    return X_aug
