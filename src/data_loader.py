"""
Data loading module for the DDR Biomarker Pipeline.

Handles reading, validation, and merging of:
  - GDSC2 drug sensitivity data (IC50 / AUC)
  - DepMap somatic mutation data
  - DepMap copy-number data
  - Cell-line metadata (tissue type, lineage)

Typical usage::

    from src.data_loader import GDSCDataLoader
    from config.config import PipelineConfig

    cfg = PipelineConfig()
    loader = GDSCDataLoader(cfg)
    gdsc2   = loader.load_gdsc2()
    muts    = loader.load_depmap_mutations()
    merged  = loader.merge_datasets(gdsc2, muts, drug="Olaparib")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema constants (expected column subsets)
# ---------------------------------------------------------------------------

_GDSC2_REQUIRED_COLS = {"DRUG_NAME", "CELL_LINE_NAME", "LN_IC50"}
_GDSC2_OPTIONAL_COLS = {"AUC", "TCGA_DESC", "COSMIC_ID", "DRUG_ID", "DATASET"}

_DEPMAP_MUT_REQUIRED_COLS = {"HugoSymbol", "ModelID"}
_DEPMAP_MUT_OPTIONAL_COLS = {
    "VariantType", "ProteinChange", "VariantAnnotation",
    "isCOSMIChotspot", "isDeleterious",
}

_DEPMAP_CN_REQUIRED_COLS = {"ModelID"}


class DataValidationError(Exception):
    """Raised when a loaded DataFrame fails schema validation."""


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------


class GDSCDataLoader:
    """
    Loads and pre-processes GDSC2 + DepMap pharmacogenomics data.

    Parameters
    ----------
    config : PipelineConfig
        Master pipeline configuration object.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._gdsc2_cache: Optional[pd.DataFrame] = None
        self._mutations_cache: Optional[pd.DataFrame] = None
        self._cn_cache: Optional[pd.DataFrame] = None
        self._meta_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_gdsc2(self, path: Optional[str] = None, force_reload: bool = False) -> pd.DataFrame:
        """
        Load the GDSC2 dose-response dataset.

        Expects a CSV with at minimum: DRUG_NAME, CELL_LINE_NAME, LN_IC50.
        GDSC2 already stores log-transformed IC50 (natural log, micromolar).

        Parameters
        ----------
        path : str, optional
            Override path from config.
        force_reload : bool
            Bypass the in-memory cache.

        Returns
        -------
        pd.DataFrame
            Columns: DRUG_NAME, CELL_LINE_NAME, LN_IC50, AUC, TCGA_DESC, ...
        """
        if self._gdsc2_cache is not None and not force_reload:
            return self._gdsc2_cache.copy()

        fpath = Path(path or self.config.data.gdsc2_ic50_path)
        logger.info("Loading GDSC2 data from %s", fpath)

        if not fpath.exists():
            raise FileNotFoundError(
                f"GDSC2 file not found: {fpath}. "
                "Download from https://www.cancerrxgene.org/downloads/bulk_download"
            )

        df = pd.read_csv(fpath, low_memory=False)
        df = self._validate_gdsc2(df)
        df = self._clean_gdsc2(df)

        self._gdsc2_cache = df
        logger.info("GDSC2 loaded: %d rows, %d unique drugs, %d unique cell lines",
                    len(df), df["DRUG_NAME"].nunique(), df["CELL_LINE_NAME"].nunique())
        return df.copy()

    def load_depmap_mutations(
        self,
        path: Optional[str] = None,
        genes: Optional[List[str]] = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Load DepMap somatic mutation calls.

        Parameters
        ----------
        path : str, optional
            Override path from config.
        genes : list of str, optional
            Subset to these gene symbols. Defaults to config DDR gene panel.
        force_reload : bool
            Bypass the in-memory cache.

        Returns
        -------
        pd.DataFrame
            Long-format table: ModelID, HugoSymbol, VariantType, isDeleterious, ...
        """
        if self._mutations_cache is not None and not force_reload:
            df = self._mutations_cache.copy()
        else:
            fpath = Path(path or self.config.data.depmap_mutations_path)
            logger.info("Loading DepMap mutation data from %s", fpath)

            if not fpath.exists():
                raise FileNotFoundError(
                    f"DepMap mutations file not found: {fpath}. "
                    "Download OmicsSomaticMutations.csv from depmap.org"
                )

            df = pd.read_csv(fpath, low_memory=False)
            df = self._validate_depmap_mutations(df)
            df = self._clean_depmap_mutations(df)
            self._mutations_cache = df

        if genes is not None:
            df = df[df["HugoSymbol"].isin(genes)].copy()

        logger.info("DepMap mutations: %d rows, %d genes, %d cell lines",
                    len(df), df["HugoSymbol"].nunique(), df["ModelID"].nunique())
        return df

    def load_depmap_copy_number(
        self, path: Optional[str] = None, genes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load DepMap gene-level copy number data.

        Returns a cell-line x gene matrix of log2(ratio) copy number values.

        Parameters
        ----------
        path : str, optional
        genes : list of str, optional
            Subset columns to these genes.

        Returns
        -------
        pd.DataFrame
            Index = ModelID, columns = gene symbols, values = log2 CN ratios.
        """
        if self._cn_cache is not None:
            df = self._cn_cache.copy()
        else:
            fpath = Path(path or self.config.data.depmap_cn_path)
            logger.info("Loading DepMap copy number data from %s", fpath)

            if not fpath.exists():
                raise FileNotFoundError(f"DepMap CN file not found: {fpath}")

            # DepMap CN files: first column is ModelID, rest are gene symbols
            df = pd.read_csv(fpath, index_col=0, low_memory=False)
            df.index.name = "ModelID"
            # Strip gene version suffixes: BRCA1 (1234) -> BRCA1
            df.columns = [c.split(" ")[0] for c in df.columns]
            self._cn_cache = df

        if genes is not None:
            available = [g for g in genes if g in df.columns]
            missing = set(genes) - set(available)
            if missing:
                logger.warning("CN data missing %d genes: %s", len(missing),
                               ", ".join(sorted(missing)[:10]))
            df = df[available]

        return df

    def load_cell_line_metadata(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load DepMap cell line metadata (tissue type, lineage, COSMIC ID).

        Returns
        -------
        pd.DataFrame
            Columns: ModelID, OncotreeLineage, OncotreePrimaryDisease,
                     SangerModelID (COSMIC ID), CellLineName
        """
        if self._meta_cache is not None:
            return self._meta_cache.copy()

        fpath = Path(path or self.config.data.cell_line_meta_path)
        if not fpath.exists():
            logger.warning("Cell line metadata not found at %s; returning empty.", fpath)
            return pd.DataFrame(columns=["ModelID", "OncotreeLineage", "CellLineName"])

        df = pd.read_csv(fpath, low_memory=False)
        self._meta_cache = df
        return df.copy()

    # ------------------------------------------------------------------
    # Computed / derived data
    # ------------------------------------------------------------------

    def compute_auc_from_ic50(self, gdsc2_df: pd.DataFrame) -> pd.DataFrame:
        """
        If AUC column is absent, approximate AUC from LN_IC50 using a sigmoidal
        dose-response model assumption. This is a surrogate: a lower IC50 maps
        to a higher AUC (more sensitive).

        AUC_approx = 1 / (1 + exp(LN_IC50 / scale))

        Parameters
        ----------
        gdsc2_df : pd.DataFrame
            GDSC2 data with LN_IC50 column.

        Returns
        -------
        pd.DataFrame
            Same DataFrame with 'AUC_approx' column added.
        """
        df = gdsc2_df.copy()
        if "AUC" in df.columns:
            df["AUC_approx"] = df["AUC"]
        else:
            scale = df["LN_IC50"].std()
            df["AUC_approx"] = 1.0 / (1.0 + np.exp(df["LN_IC50"] / max(scale, 1e-6)))
        return df

    def get_cell_lines_by_tissue(
        self, gdsc2_df: pd.DataFrame, tissue: str
    ) -> pd.DataFrame:
        """
        Subset GDSC2 data to a specific TCGA tissue type.

        Parameters
        ----------
        gdsc2_df : pd.DataFrame
        tissue : str
            TCGA tissue abbreviation, e.g. "BRCA", "OV".

        Returns
        -------
        pd.DataFrame
        """
        if "TCGA_DESC" not in gdsc2_df.columns:
            logger.warning("TCGA_DESC column not found; cannot filter by tissue.")
            return gdsc2_df

        subset = gdsc2_df[gdsc2_df["TCGA_DESC"].str.upper() == tissue.upper()]
        logger.info("Tissue '%s': %d rows, %d cell lines",
                    tissue, len(subset), subset["CELL_LINE_NAME"].nunique())
        return subset

    def merge_datasets(
        self,
        gdsc2_df: pd.DataFrame,
        mutations_df: pd.DataFrame,
        drug: str,
        cn_df: Optional[pd.DataFrame] = None,
        meta_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge GDSC2 drug response with DepMap mutation (and optionally CN) data.

        Steps:
        1. Filter GDSC2 to the specified drug.
        2. Pivot mutations to a wide binary matrix (cell line x gene).
        3. Join on cell line name / ModelID.
        4. Optionally join copy number features.

        Parameters
        ----------
        gdsc2_df : pd.DataFrame
            Full GDSC2 table.
        mutations_df : pd.DataFrame
            DepMap long-format mutation table.
        drug : str
            Drug name to filter (case-insensitive partial match).
        cn_df : pd.DataFrame, optional
            Copy number matrix (ModelID x gene).
        meta_df : pd.DataFrame, optional
            Cell line metadata with tissue information.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (merged_df, drug_ic50_series)
            merged_df has columns: cell_line, LN_IC50, [mutation features], [CN features]
            drug_ic50_series is just the IC50 vector for quick downstream use.
        """
        # --- Filter to drug ---
        canonical = self.config.resolve_drug_name(drug)
        mask = gdsc2_df["DRUG_NAME"].str.lower() == canonical.lower()
        if mask.sum() == 0:
            # Try partial match
            mask = gdsc2_df["DRUG_NAME"].str.lower().str.contains(
                drug.lower(), regex=False
            )
        drug_df = gdsc2_df[mask][["CELL_LINE_NAME", "LN_IC50", "TCGA_DESC"]].copy()

        if len(drug_df) == 0:
            raise ValueError(
                f"No data found for drug '{drug}'. "
                f"Available drugs: {sorted(gdsc2_df['DRUG_NAME'].unique())[:20]}"
            )
        logger.info("Drug '%s': %d cell lines with IC50 data", drug, len(drug_df))

        # --- Pivot mutations to binary matrix ---
        lof_classes = set(self.config.data.depmap_lof_classes)
        lof_muts = mutations_df[
            mutations_df["VariantType"].isin(lof_classes)
        ] if "VariantType" in mutations_df.columns else mutations_df

        # Create binary LoF matrix: ModelID x gene
        mut_matrix = (
            lof_muts.groupby(["ModelID", "HugoSymbol"])
            .size()
            .gt(0)
            .astype(int)
            .unstack(fill_value=0)
        )
        mut_matrix.columns.name = None
        mut_matrix = mut_matrix.reset_index()
        mut_matrix = mut_matrix.rename(columns={"ModelID": "CELL_LINE_NAME"})

        # Rename gene columns to avoid clashes
        gene_cols = [c for c in mut_matrix.columns if c != "CELL_LINE_NAME"]
        rename_map = {g: f"{g}_lof_mutation" for g in gene_cols}
        mut_matrix = mut_matrix.rename(columns=rename_map)

        # --- Normalize cell line names ---
        drug_df["CELL_LINE_NAME"] = drug_df["CELL_LINE_NAME"].str.strip()
        mut_matrix["CELL_LINE_NAME"] = mut_matrix["CELL_LINE_NAME"].str.strip()

        # --- Merge drug response + mutations ---
        merged = drug_df.merge(mut_matrix, on="CELL_LINE_NAME", how="inner")

        # --- Optionally join CN features ---
        if cn_df is not None:
            cn_subset = cn_df.copy()
            cn_subset.index.name = "CELL_LINE_NAME"
            cn_subset = cn_subset.reset_index()
            cn_subset.columns = [
                "CELL_LINE_NAME" if c == "CELL_LINE_NAME"
                else f"{c}_cn"
                for c in cn_subset.columns
            ]
            merged = merged.merge(cn_subset, on="CELL_LINE_NAME", how="left")

        # --- Optionally join metadata ---
        if meta_df is not None and "CellLineName" in meta_df.columns:
            meta_slim = meta_df[["CellLineName", "OncotreeLineage"]].copy()
            meta_slim = meta_slim.rename(columns={"CellLineName": "CELL_LINE_NAME"})
            merged = merged.merge(meta_slim, on="CELL_LINE_NAME", how="left")

        n_missing = drug_df.shape[0] - merged.shape[0]
        logger.info(
            "Merged dataset: %d cell lines (%d dropped — no mutation overlap)",
            merged.shape[0], n_missing,
        )

        ic50_series = merged.set_index("CELL_LINE_NAME")["LN_IC50"]
        return merged, ic50_series

    def get_drug_ic50_matrix(
        self, gdsc2_df: pd.DataFrame, drugs: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Return a cell-line x drug IC50 matrix (LN_IC50).

        Parameters
        ----------
        gdsc2_df : pd.DataFrame
        drugs : list of str, optional
            Subset to these drugs. If None, uses all DDR drugs.

        Returns
        -------
        pd.DataFrame
            Index = CELL_LINE_NAME, columns = DRUG_NAME, values = LN_IC50.
        """
        if drugs is None:
            drugs = self.config.drugs

        canonical_drugs = [self.config.resolve_drug_name(d) for d in drugs]
        subset = gdsc2_df[gdsc2_df["DRUG_NAME"].isin(canonical_drugs)]

        pivot = subset.pivot_table(
            index="CELL_LINE_NAME", columns="DRUG_NAME",
            values="LN_IC50", aggfunc="mean"
        )
        logger.info("IC50 matrix: %d cell lines x %d drugs", *pivot.shape)
        return pivot

    # ------------------------------------------------------------------
    # Private validation / cleaning helpers
    # ------------------------------------------------------------------

    def _validate_gdsc2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate GDSC2 DataFrame has required columns."""
        missing = _GDSC2_REQUIRED_COLS - set(df.columns)
        if missing:
            raise DataValidationError(
                f"GDSC2 file missing required columns: {missing}. "
                f"Found: {list(df.columns[:10])}"
            )
        return df

    def _clean_gdsc2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize and clean GDSC2 data."""
        df = df.copy()
        df["DRUG_NAME"] = df["DRUG_NAME"].str.strip()
        df["CELL_LINE_NAME"] = df["CELL_LINE_NAME"].str.strip()
        # Drop rows with null IC50
        n_before = len(df)
        df = df.dropna(subset=["LN_IC50"])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.warning("Dropped %d rows with null LN_IC50", n_dropped)

        # Cap extreme IC50 values (>5 SD from mean are likely assay failures)
        ic50_mean = df["LN_IC50"].mean()
        ic50_std = df["LN_IC50"].std()
        cap_low = ic50_mean - 5 * ic50_std
        cap_high = ic50_mean + 5 * ic50_std
        n_capped = ((df["LN_IC50"] < cap_low) | (df["LN_IC50"] > cap_high)).sum()
        if n_capped > 0:
            logger.warning("Capping %d extreme LN_IC50 values (±5 SD)", n_capped)
        df["LN_IC50"] = df["LN_IC50"].clip(cap_low, cap_high)
        return df

    def _validate_depmap_mutations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate DepMap mutation DataFrame."""
        missing = _DEPMAP_MUT_REQUIRED_COLS - set(df.columns)
        if missing:
            raise DataValidationError(
                f"DepMap mutation file missing required columns: {missing}. "
                f"Found: {list(df.columns[:10])}"
            )
        return df

    def _clean_depmap_mutations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DepMap mutation data."""
        df = df.copy()
        df["HugoSymbol"] = df["HugoSymbol"].str.strip()
        df["ModelID"] = df["ModelID"].str.strip()
        # Remove null gene symbols
        df = df.dropna(subset=["HugoSymbol"])
        df = df[df["HugoSymbol"] != ""]
        return df

    def summarize_drug_coverage(self, gdsc2_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a summary table of drug coverage in GDSC2.

        Returns
        -------
        pd.DataFrame
            Columns: DRUG_NAME, n_cell_lines, mean_LN_IC50, std_LN_IC50
        """
        summary = (
            gdsc2_df.groupby("DRUG_NAME")["LN_IC50"]
            .agg(n_cell_lines="count", mean_LN_IC50="mean", std_LN_IC50="std")
            .reset_index()
            .sort_values("n_cell_lines", ascending=False)
        )
        return summary

    def generate_synthetic_data(
        self,
        n_cell_lines: int = 200,
        n_genes: int = 20,
        drug: str = "Olaparib",
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic GDSC2 + DepMap mutation data for testing.

        Injects a signal: BRCA1/BRCA2 LoF mutations are associated with
        lower IC50 (sensitivity) to olaparib-like drugs.

        Parameters
        ----------
        n_cell_lines : int
        n_genes : int
        drug : str
        seed : int

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (gdsc2_df, mutations_df)
        """
        rng = np.random.default_rng(seed)

        gene_panel = self.config.ddr_genes[:n_genes]
        cell_lines = [f"CL_{i:04d}" for i in range(n_cell_lines)]

        # --- GDSC2 IC50 data ---
        # Base IC50: normal distribution
        base_ic50 = rng.normal(2.0, 1.5, n_cell_lines)

        # Inject BRCA1/2 signal: ~15% of cell lines are BRCA-mutant
        brca_mutant = rng.random(n_cell_lines) < 0.15
        base_ic50[brca_mutant] -= 2.5  # Lower IC50 = more sensitive

        gdsc2_df = pd.DataFrame({
            "DRUG_NAME": drug,
            "CELL_LINE_NAME": cell_lines,
            "LN_IC50": base_ic50,
            "AUC": np.clip(1.0 / (1.0 + np.exp(base_ic50 / 1.5)), 0.01, 0.99),
            "TCGA_DESC": rng.choice(
                ["BRCA", "OV", "LUAD", "COAD", "STAD"], n_cell_lines
            ),
        })

        # --- DepMap mutation data ---
        mut_records = []
        lof_types = self.config.data.depmap_lof_classes

        for i, cl in enumerate(cell_lines):
            for gene in gene_panel:
                # Background mutation rate: ~5% per gene
                mutation_prob = 0.05
                # Elevated rate for BRCA1/2 in BRCA-mutant cell lines
                if brca_mutant[i] and gene in ("BRCA1", "BRCA2"):
                    mutation_prob = 0.7

                if rng.random() < mutation_prob:
                    mut_records.append({
                        "ModelID": cl,
                        "HugoSymbol": gene,
                        "VariantType": rng.choice(lof_types),
                        "ProteinChange": f"p.X{rng.integers(100, 3000)}*",
                    })

        mutations_df = pd.DataFrame(mut_records)
        return gdsc2_df, mutations_df
