"""
Centralized configuration for the DDR Biomarker & Drug Response Prediction Pipeline.

All paths, hyperparameters, gene lists, and pipeline settings live here.
Import PipelineConfig at the top of any module that needs configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Paths and schema settings for input data."""

    # Input data paths (override via environment variables or constructor kwargs)
    gdsc2_ic50_path: str = os.environ.get(
        "GDSC2_IC50_PATH", "data/GDSC2_fitted_dose_response.csv"
    )
    depmap_mutations_path: str = os.environ.get(
        "DEPMAP_MUTATIONS_PATH", "data/OmicsSomaticMutations.csv"
    )
    depmap_cn_path: str = os.environ.get(
        "DEPMAP_CN_PATH", "data/OmicsCNGene.csv"
    )
    depmap_expression_path: str = os.environ.get(
        "DEPMAP_EXPR_PATH", "data/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    )
    cell_line_meta_path: str = os.environ.get(
        "CELL_LINE_META_PATH", "data/Model.csv"
    )

    # Output directories
    output_dir: str = "results"
    model_dir: str = "results/models"
    plot_dir: str = "results/plots"
    report_dir: str = "results/reports"

    # GDSC2 column names
    gdsc2_drug_col: str = "DRUG_NAME"
    gdsc2_cell_line_col: str = "CELL_LINE_NAME"
    gdsc2_cosmic_col: str = "COSMIC_ID"
    gdsc2_ic50_col: str = "LN_IC50"
    gdsc2_auc_col: str = "AUC"
    gdsc2_tissue_col: str = "TCGA_DESC"

    # DepMap column names
    depmap_gene_col: str = "HugoSymbol"
    depmap_cell_line_col: str = "ModelID"
    depmap_var_class_col: str = "VariantType"
    depmap_protein_change_col: str = "ProteinChange"
    depmap_lof_classes: List[str] = field(default_factory=lambda: [
        "Nonsense_Mutation",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "Splice_Site",
        "Translation_Start_Site",
        "Nonstop_Mutation",
    ])

    # IC50 transformation
    ic50_log_transform: bool = True  # GDSC2 already provides LN_IC50
    ic50_z_score_within_tissue: bool = True


@dataclass
class ModelConfig:
    """Hyperparameters for each model in the ensemble."""

    # Logistic Regression
    logistic_regression: Dict = field(default_factory=lambda: {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    })

    # Random Forest
    random_forest: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    })

    # Gradient Boosting
    gradient_boosting: Dict = field(default_factory=lambda: {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.8,
        "min_samples_leaf": 5,
        "random_state": 42,
    })

    # Elastic Net (used as a linear classifier via SGDClassifier)
    elastic_net: Dict = field(default_factory=lambda: {
        "alpha": 0.01,
        "l1_ratio": 0.5,
        "max_iter": 2000,
        "tol": 1e-4,
        "class_weight": "balanced",
        "random_state": 42,
    })


@dataclass
class CVConfig:
    """Cross-validation settings."""

    n_folds: int = 5
    shuffle: bool = True
    random_state: int = 42
    scoring_metric: str = "roc_auc"

    # Sensitivity threshold for binarizing IC50 (fraction below median = sensitive)
    sensitivity_quantile: float = 0.5


@dataclass
class ShapConfig:
    """SHAP explainability settings."""

    max_samples: int = 200          # Max background samples for SHAP explainer
    top_n_features: int = 20        # Number of top features to display/report
    use_tree_explainer: bool = True  # Use TreeExplainer for RF/GB; Linear for LR/EN
    plot_summary: bool = True
    plot_dependence: bool = True
    save_shap_values: bool = True


# ---------------------------------------------------------------------------
# Drug lists
# ---------------------------------------------------------------------------

PARP_INHIBITORS: List[str] = [
    "olaparib",
    "rucaparib",
    "niraparib",
    "talazoparib",
    "veliparib",
]

ATR_INHIBITORS: List[str] = [
    "AZD6738",    # ceralasertib
    "VE-822",     # berzosertib
    "AZD6738",
    "M6620",
]

OTHER_DDR_DRUGS: List[str] = [
    "5-Fluorouracil",
    "gemcitabine",
    "camptothecin",
    "topotecan",
]

ALL_DDR_DRUGS: List[str] = PARP_INHIBITORS + ATR_INHIBITORS + OTHER_DDR_DRUGS

# Canonical drug name aliases (lowercase -> GDSC2 name)
DRUG_NAME_ALIASES: Dict[str, str] = {
    "olaparib": "Olaparib",
    "rucaparib": "Rucaparib",
    "niraparib": "Niraparib",
    "talazoparib": "Talazoparib",
    "veliparib": "Veliparib",
    "azd6738": "AZD6738",
    "ve-822": "VE-822",
    "5-fluorouracil": "5-Fluorouracil",
    "gemcitabine": "Gemcitabine",
    "camptothecin": "Camptothecin",
}

# ---------------------------------------------------------------------------
# Gene lists
# ---------------------------------------------------------------------------

# Core DDR gene panel used for feature engineering
DDR_GENES: List[str] = [
    # Homologous Recombination
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "BRIP1",
    "BARD1", "NBN", "MRE11", "RAD50",
    # ATM/ATR signaling
    "ATM", "ATR", "CHEK1", "CHEK2", "TP53BP1",
    # Mismatch Repair
    "MSH2", "MSH6", "MLH1", "PMS2", "MSH3",
    # ARID1A / SWI-SNF
    "ARID1A", "ARID1B", "SMARCA4", "SMARCB1",
    # Other DDR
    "PARP1", "PARP2", "XRCC1", "XRCC2", "XRCC3",
    "FANCA", "FANCC", "FANCD2", "FANCG", "FANCI",
    "CDK12", "CCNE1", "RB1", "TP53",
]

# HR deficiency (HRD) genes for composite HRD score
HRD_GENES: List[str] = [
    "BRCA1", "BRCA2", "PALB2", "RAD51C", "RAD51D", "BRIP1",
    "BARD1", "NBN",
]

# Mismatch repair (MMR) genes for MSI classification
MMR_GENES: List[str] = ["MSH2", "MSH6", "MLH1", "PMS2", "MSH3"]

# ATR inhibitor-relevant genes
ATR_RELEVANT_GENES: List[str] = [
    "ARID1A", "ATM", "CCNE1", "CDK12", "RB1", "TP53",
    "MSH2", "MLH1",
]

# GDSC2 tissue type labels
GDSC2_TISSUE_TYPES: List[str] = [
    "BRCA",    # Breast cancer
    "OV",      # Ovarian cancer
    "PRAD",    # Prostate cancer
    "PAAD",    # Pancreatic cancer
    "COAD",    # Colorectal cancer
    "STAD",    # Gastric cancer
    "LUAD",    # Lung adenocarcinoma
    "LUSC",    # Lung squamous
    "SKCM",    # Melanoma
    "GBM",     # Glioblastoma
    "BLCA",    # Bladder cancer
    "HNSC",    # Head and neck
    "UCEC",    # Endometrial cancer
    "KIRC",    # Kidney clear cell
    "THCA",    # Thyroid
    "LAML",    # AML
    "DLBC",    # Diffuse large B-cell lymphoma
]


# ---------------------------------------------------------------------------
# Master pipeline config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """
    Master configuration object for the DDR Biomarker Pipeline.

    Usage::

        from config.config import PipelineConfig
        cfg = PipelineConfig()
        cfg.drugs           # list of drugs to analyze
        cfg.ddr_genes       # DDR gene panel
        cfg.model.random_forest  # model hyperparameters
        cfg.cv.n_folds      # cross-validation folds
    """

    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)

    # Convenience accessors (mirrors sub-config fields for backwards compat)
    @property
    def drugs(self) -> List[str]:
        return PARP_INHIBITORS[:3] + ATR_INHIBITORS[:2] + ["5-Fluorouracil"]

    @property
    def parp_inhibitors(self) -> List[str]:
        return PARP_INHIBITORS

    @property
    def atr_inhibitors(self) -> List[str]:
        return ATR_INHIBITORS

    @property
    def ddr_genes(self) -> List[str]:
        return DDR_GENES

    @property
    def hrd_genes(self) -> List[str]:
        return HRD_GENES

    @property
    def mmr_genes(self) -> List[str]:
        return MMR_GENES

    @property
    def tissue_types(self) -> List[str]:
        return GDSC2_TISSUE_TYPES

    @property
    def cv_folds(self) -> int:
        return self.cv.n_folds

    @property
    def model_params(self) -> Dict[str, Dict]:
        """Return all model hyperparameter dicts."""
        return {
            "logistic_regression": self.model.logistic_regression,
            "random_forest": self.model.random_forest,
            "gradient_boosting": self.model.gradient_boosting,
            "elastic_net": self.model.elastic_net,
        }

    def get_output_path(self, drug: str, subdir: str = "") -> Path:
        """Return output path for a given drug analysis."""
        base = Path(self.data.output_dir) / drug.lower().replace(" ", "_")
        if subdir:
            base = base / subdir
        base.mkdir(parents=True, exist_ok=True)
        return base

    def resolve_drug_name(self, drug: str) -> str:
        """Resolve canonical GDSC2 drug name from alias."""
        return DRUG_NAME_ALIASES.get(drug.lower(), drug)
