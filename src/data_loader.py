"""
Data loading and synthetic data generation for the DDR Biomarker Pipeline.

Provides helpers to generate realistic pharmacogenomic datasets when real
GDSC2 / DepMap files are not available, plus loaders for the real formats.
"""

import numpy as np
import pandas as pd

from .utils import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    n_cell_lines: int = 200,
    n_drugs: int = 5,
    seed: int = 42,
    ddr_genes: list | None = None,
    drug_names: list | None = None,
    tissue_types: list | None = None,
    mutation_rate: float = 0.10,
    sensitivity_boost: float = 0.30,
) -> dict[str, pd.DataFrame]:
    """Generate a fully synthetic pharmacogenomic dataset.

    The generator mimics the joint distribution observed in real GDSC2 data:
    cell lines harbouring DDR mutations tend to show lower IC50 values for
    PARPi / ATRi compounds.

    Args:
        n_cell_lines: Number of synthetic cell lines to create.
        n_drugs: Number of drugs (ignored when *drug_names* is given).
        seed: Random seed for reproducibility.
        ddr_genes: Gene symbol list; defaults to the standard 13-gene panel.
        drug_names: Drug name list; defaults to the DDR drug panel.
        tissue_types: Tissue labels for metadata.
        mutation_rate: Per-gene mutation probability.
        sensitivity_boost: Extra IC50 shift (negative) for DDR-mutant lines.

    Returns:
        Dictionary with keys ``"ic50"``, ``"mutations"``, ``"metadata"``,
        each mapping to a :class:`pandas.DataFrame`.
    """
    rng = np.random.RandomState(seed)

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
    if drug_names is None:
        drug_names = ["olaparib", "rucaparib", "talazoparib", "AZD6738", "VE-822"]
    if tissue_types is None:
        tissue_types = [
            "breast",
            "ovarian",
            "prostate",
            "lung",
            "pancreatic",
            "colorectal",
            "bladder",
            "endometrial",
        ]

    n_drugs = len(drug_names)
    cell_line_ids = [f"CL_{i:04d}" for i in range(n_cell_lines)]

    # ------------------------------------------------------------------
    # 1. Mutation matrix  (cell_lines x genes), binary, ~mutation_rate
    # ------------------------------------------------------------------
    mutation_matrix = (rng.rand(n_cell_lines, len(ddr_genes)) < mutation_rate).astype(int)
    mutation_df = pd.DataFrame(
        mutation_matrix,
        index=cell_line_ids,
        columns=ddr_genes,
    )
    mutation_df.index.name = "cell_line_id"

    # ------------------------------------------------------------------
    # 2. IC50 matrix  (cell_lines x drugs), log-normal baseline
    # ------------------------------------------------------------------
    # Base IC50 drawn from a log-normal centred at 0 (log scale)
    base_ic50 = rng.normal(loc=0.5, scale=1.0, size=(n_cell_lines, n_drugs))

    # DDR-mutant lines get lower IC50 for PARPi/ATRi drugs
    hr_genes = ["BRCA1", "BRCA2", "PALB2", "RAD51", "ATM", "ATR"]
    hr_gene_idx = [ddr_genes.index(g) for g in hr_genes if g in ddr_genes]
    hr_burden = mutation_matrix[:, hr_gene_idx].sum(axis=1)  # 0..6

    for drug_idx in range(n_drugs):
        # Each drug has a slightly different effect magnitude
        drug_effect = sensitivity_boost * (1.0 + 0.2 * rng.randn())
        base_ic50[:, drug_idx] -= drug_effect * hr_burden
        # Add per-drug noise
        base_ic50[:, drug_idx] += rng.normal(0, 0.3, size=n_cell_lines)

    ic50_df = pd.DataFrame(
        base_ic50,
        index=cell_line_ids,
        columns=drug_names,
    )
    ic50_df.index.name = "cell_line_id"

    # ------------------------------------------------------------------
    # 3. Metadata
    # ------------------------------------------------------------------
    tissues = rng.choice(tissue_types, size=n_cell_lines)
    msi_status = (mutation_df[["MLH1", "MSH2", "MSH6"]].sum(axis=1) > 0).astype(int).values

    metadata_df = pd.DataFrame(
        {
            "cell_line_id": cell_line_ids,
            "cell_line_name": [f"CellLine-{i}" for i in range(n_cell_lines)],
            "tissue": tissues,
            "msi_status": msi_status,
        }
    ).set_index("cell_line_id")

    logger.info(
        "Generated synthetic data: %d cell lines, %d drugs, %d genes",
        n_cell_lines,
        n_drugs,
        len(ddr_genes),
    )

    return {
        "ic50": ic50_df,
        "mutations": mutation_df,
        "metadata": metadata_df,
    }


# ---------------------------------------------------------------------------
# Real-data loaders (fall back to synthetic)
# ---------------------------------------------------------------------------


def load_gdsc2_data(path: str | None = None) -> pd.DataFrame:
    """Load GDSC2 IC50 data from *path*.

    If the file does not exist or *path* is ``None``, synthetic IC50 data is
    returned instead so the pipeline can always run.

    Args:
        path: Path to a CSV with cell-line IDs as rows and drugs as columns.

    Returns:
        IC50 :class:`~pandas.DataFrame`.
    """
    if path is not None:
        try:
            df = pd.read_csv(path, index_col=0)
            logger.info("Loaded GDSC2 data from %s (%s)", path, df.shape)
            return df
        except FileNotFoundError:
            logger.warning("GDSC2 file not found at %s \u2014 using synthetic data", path)

    return generate_synthetic_data()["ic50"]


def load_depmap_mutations(path: str | None = None) -> pd.DataFrame:
    """Load DepMap mutation calls from *path*.

    If the file does not exist or *path* is ``None``, synthetic mutation data
    is returned instead.

    Args:
        path: Path to a CSV with cell-line IDs as rows, genes as columns.

    Returns:
        Binary mutation :class:`~pandas.DataFrame`.
    """
    if path is not None:
        try:
            df = pd.read_csv(path, index_col=0)
            logger.info("Loaded DepMap mutations from %s (%s)", path, df.shape)
            return df
        except FileNotFoundError:
            logger.warning("DepMap file not found at %s \u2014 using synthetic data", path)

    return generate_synthetic_data()["mutations"]


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_datasets(
    ic50_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Inner-join IC50 and mutation matrices on their shared cell-line index.

    Args:
        ic50_df: IC50 values, rows = cell lines, columns = drugs.
        mutation_df: Binary mutation calls, rows = cell lines, columns = genes.
        metadata_df: Optional metadata to join.

    Returns:
        Merged :class:`~pandas.DataFrame` containing both IC50 and mutation
        columns for cell lines present in **both** inputs.
    """
    common = ic50_df.index.intersection(mutation_df.index)
    logger.info(
        "Merging datasets: %d IC50 lines, %d mutation lines, %d in common",
        len(ic50_df),
        len(mutation_df),
        len(common),
    )

    merged = ic50_df.loc[common].join(mutation_df.loc[common], rsuffix="_mut")

    if metadata_df is not None:
        meta_common = common.intersection(metadata_df.index)
        merged = merged.loc[meta_common].join(metadata_df.loc[meta_common], rsuffix="_meta")

    return merged
