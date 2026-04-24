"""
Pipeline configuration for the DDR Biomarker Discovery Pipeline.

Defines gene lists, model hyperparameters, drug panels, and all
configurable constants used across the pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Central configuration for the DDR biomarker pipeline.

    Holds gene lists, model hyper-parameters, file paths, drug metadata,
    and every tuneable constant so that experiments are fully reproducible
    from a single object.

    Attributes:
        DDR_GENES: All DNA-damage-repair genes tracked in this pipeline.
        HR_GENES: Subset used for Homologous Recombination Deficiency scoring.
        MMR_GENES: Mismatch-repair genes used for MSI status.
        MODEL_PARAMS: Nested dict of sklearn-compatible hyper-parameter grids.
        RANDOM_SEED: Global seed for reproducibility.
        N_FOLDS: Number of cross-validation folds.
        OUTPUT_DIR: Where results, plots, and tables are written.
        DATA_DIR: Where raw / processed data files live.
        DDR_DRUGS: Drug names relevant to DDR synthetic-lethality screens.
        IC50_THRESHOLD: Log-IC50 threshold; below = sensitive.
        TISSUE_TYPES: Tissue types used in synthetic data generation.
        MUTATION_RATE: Background per-gene mutation probability.
        SENSITIVITY_BOOST: Extra sensitivity probability for DDR-mutant lines.
        MIN_SAMPLES_PER_CLASS: Minimum class size to proceed with modelling.
    """

    # ------------------------------------------------------------------
    # Gene panels
    # ------------------------------------------------------------------
    DDR_GENES: list[str] = field(
        default_factory=lambda: [
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
    )

    HR_GENES: list[str] = field(
        default_factory=lambda: [
            "BRCA1",
            "BRCA2",
            "PALB2",
            "RAD51",
            "ATM",
            "ATR",
        ]
    )

    MMR_GENES: list[str] = field(
        default_factory=lambda: [
            "MLH1",
            "MSH2",
            "MSH6",
        ]
    )

    # ------------------------------------------------------------------
    # Model hyper-parameters
    # ------------------------------------------------------------------
    MODEL_PARAMS: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "LogisticRegression": {
                "C_values": [0.01, 0.1, 1.0, 10.0],
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 2000,
                "class_weight": "balanced",
            },
            "GradientBoosting": {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
            },
        }
    )

    # ------------------------------------------------------------------
    # Reproducibility & cross-validation
    # ------------------------------------------------------------------
    RANDOM_SEED: int = 42
    N_FOLDS: int = 5

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    OUTPUT_DIR: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
        )
    )
    DATA_DIR: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        )
    )

    # ------------------------------------------------------------------
    # Drug panel
    # ------------------------------------------------------------------
    DDR_DRUGS: list[str] = field(
        default_factory=lambda: [
            "olaparib",
            "rucaparib",
            "talazoparib",
            "AZD6738",
            "VE-822",
        ]
    )

    # ------------------------------------------------------------------
    # IC50 classification
    # ------------------------------------------------------------------
    IC50_THRESHOLD: float = 0.0  # log-IC50; values below are "sensitive"

    # ------------------------------------------------------------------
    # Synthetic-data parameters
    # ------------------------------------------------------------------
    TISSUE_TYPES: list[str] = field(
        default_factory=lambda: [
            "breast",
            "ovarian",
            "prostate",
            "lung",
            "pancreatic",
            "colorectal",
            "bladder",
            "endometrial",
        ]
    )

    MUTATION_RATE: float = 0.10
    SENSITIVITY_BOOST: float = 0.30
    MIN_SAMPLES_PER_CLASS: int = 15

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_output_path(self, filename: str) -> str:
        """Return a full path inside OUTPUT_DIR for *filename*."""
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        return os.path.join(self.OUTPUT_DIR, filename)

    def get_data_path(self, filename: str) -> str:
        """Return a full path inside DATA_DIR for *filename*."""
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)
        return os.path.join(self.DATA_DIR, filename)

    def get_lr_param_grid(self) -> list[dict[str, Any]]:
        """Return an sklearn-compatible parameter grid for LogisticRegression."""
        lr = self.MODEL_PARAMS["LogisticRegression"]
        return [
            {
                "C": c,
                "penalty": lr["penalty"],
                "solver": lr["solver"],
                "max_iter": lr["max_iter"],
                "class_weight": lr["class_weight"],
            }
            for c in lr["C_values"]
        ]

    def get_gb_params(self) -> dict[str, Any]:
        """Return a flat dict of GradientBoosting constructor kwargs."""
        return dict(self.MODEL_PARAMS["GradientBoosting"])

    def validate(self) -> None:
        """Run basic sanity checks on the configuration.

        Raises:
            ValueError: If any check fails.
        """
        if not set(self.HR_GENES).issubset(set(self.DDR_GENES)):
            raise ValueError("HR_GENES must be a subset of DDR_GENES")
        if not set(self.MMR_GENES).issubset(set(self.DDR_GENES)):
            raise ValueError("MMR_GENES must be a subset of DDR_GENES")
        if self.N_FOLDS < 2:
            raise ValueError("N_FOLDS must be >= 2")
        if self.RANDOM_SEED < 0:
            raise ValueError("RANDOM_SEED must be non-negative")
        if not self.DDR_DRUGS:
            raise ValueError("DDR_DRUGS must not be empty")
        if not (0 < self.MUTATION_RATE < 1):
            raise ValueError("MUTATION_RATE must be in (0, 1)")

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "DDR Biomarker Pipeline Configuration",
            "=" * 40,
            f"  DDR genes       : {len(self.DDR_GENES)}",
            f"  HR genes        : {len(self.HR_GENES)}",
            f"  MMR genes       : {len(self.MMR_GENES)}",
            f"  Drugs           : {', '.join(self.DDR_DRUGS)}",
            f"  IC50 threshold  : {self.IC50_THRESHOLD}",
            f"  Random seed     : {self.RANDOM_SEED}",
            f"  CV folds        : {self.N_FOLDS}",
            f"  Output dir      : {self.OUTPUT_DIR}",
            f"  Data dir        : {self.DATA_DIR}",
            f"  Mutation rate   : {self.MUTATION_RATE}",
            f"  Min class size  : {self.MIN_SAMPLES_PER_CLASS}",
        ]
        return "\n".join(lines)

    def __post_init__(self) -> None:
        """Ensure output directories exist after construction."""
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
