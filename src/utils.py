"""
Utility functions for the DDR Biomarker Pipeline.

Provides shared helpers for:
  - Logging setup (console + file handlers)
  - Configuration loading
  - File I/O (results saving, directory creation)
  - Statistical helpers (confidence intervals, p-value formatting)
  - Stratified train/test splitting

Typical usage::

    from src.utils import setup_logging, save_results, stratified_split

    setup_logging(log_level="INFO", log_file="results/run.log")
    save_results(metrics_dict, output_dir="results/olaparib")
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure the root logger with console (and optionally file) handlers.

    Parameters
    ----------
    log_level : str
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file : str, optional
        If provided, also write logs to this file path.
    fmt : str
        Log format string.
    date_fmt : str
        Date format for log timestamps.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate messages
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info("Logging to file: %s", log_path)

    root_logger.info("Logging initialized at level %s", log_level.upper())
    return root_logger


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def load_config(config_path: Optional[str] = None):
    """
    Load pipeline configuration.

    If config_path is provided, loads a JSON override file and applies
    overrides to the default PipelineConfig. Otherwise returns defaults.

    Parameters
    ----------
    config_path : str, optional
        Path to a JSON config file with override values.

    Returns
    -------
    PipelineConfig
    """
    from config.config import PipelineConfig

    cfg = PipelineConfig()

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            overrides = json.load(f)

        # Apply top-level overrides to DataConfig
        data_overrides = overrides.get("data", {})
        for key, value in data_overrides.items():
            if hasattr(cfg.data, key):
                setattr(cfg.data, key, value)

        # Apply CV overrides
        cv_overrides = overrides.get("cv", {})
        for key, value in cv_overrides.items():
            if hasattr(cfg.cv, key):
                setattr(cfg.cv, key, value)

        logger.info("Applied config overrides from %s", config_path)

    return cfg


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Create a directory (and all parents) if it does not exist.

    Parameters
    ----------
    directory : str or Path

    Returns
    -------
    Path
        The resolved directory path.
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "results.json",
    also_save_csv: bool = True,
) -> Path:
    """
    Save a results dictionary to JSON (and optionally CSV).

    Parameters
    ----------
    results : dict
        Metrics or any serializable dict.
    output_dir : str or Path
    filename : str
        Name of the JSON output file.
    also_save_csv : bool
        If True and results can be converted to a DataFrame, also save CSV.

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    out_dir = ensure_dir(output_dir)
    json_path = out_dir / filename

    # Convert numpy types for JSON serialization
    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)

    logger.info("Results saved: %s", json_path)

    if also_save_csv:
        try:
            flat = {k: v for k, v in results.items() if not isinstance(v, (dict, list))}
            if flat:
                csv_path = out_dir / filename.replace(".json", ".csv")
                pd.DataFrame([flat]).to_csv(csv_path, index=False)
        except Exception:
            pass

    return json_path


def save_dataframe(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    filename: str,
    index: bool = False,
) -> Path:
    """
    Save a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : str or Path
    filename : str
    index : bool

    Returns
    -------
    Path
    """
    out_dir = ensure_dir(output_dir)
    path = out_dir / filename
    df.to_csv(path, index=index)
    logger.info("DataFrame saved: %s (%d rows x %d cols)", path, *df.shape)
    return path


def save_pickle(obj: Any, path: Union[str, Path]) -> Path:
    """Serialize an arbitrary object to disk using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Object saved: %s", path)
    return path


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a pickled object from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
    method: str = "t",
) -> Tuple[float, float]:
    """
    Compute confidence interval for the mean of an array.

    Parameters
    ----------
    values : array-like
    confidence : float
        Confidence level (default: 0.95 for 95% CI).
    method : str
        't' for t-distribution CI (default), 'bootstrap' for bootstrap CI.

    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 2:
        return (float(values[0]), float(values[0])) if n == 1 else (np.nan, np.nan)

    if method == "t":
        mean = values.mean()
        se = stats.sem(values)
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_crit * se
        return (float(mean - margin), float(mean + margin))

    elif method == "bootstrap":
        rng = np.random.default_rng(42)
        n_boot = 10_000
        boot_means = [
            rng.choice(values, size=n, replace=True).mean()
            for _ in range(n_boot)
        ]
        alpha = (1 - confidence) / 2
        lower = np.percentile(boot_means, 100 * alpha)
        upper = np.percentile(boot_means, 100 * (1 - alpha))
        return (float(lower), float(upper))

    else:
        raise ValueError(f"Unknown CI method: {method}")


def format_pvalue(
    pval: float,
    significance_levels: Optional[Dict[float, str]] = None,
) -> str:
    """
    Format a p-value for display.

    Returns scientific notation for small values, includes significance stars.

    Parameters
    ----------
    pval : float
    significance_levels : dict, optional
        Mapping threshold -> star string. Defaults to standard thresholds.

    Returns
    -------
    str
        Formatted p-value string, e.g. "p=3.2e-05 **"
    """
    if significance_levels is None:
        significance_levels = {0.001: "***", 0.01: "**", 0.05: "*"}

    if np.isnan(pval):
        return "p=NA"

    stars = ""
    for threshold, star in sorted(significance_levels.items()):
        if pval < threshold:
            stars = star
            break

    if pval < 0.001:
        pval_str = f"p={pval:.2e}"
    elif pval < 0.05:
        pval_str = f"p={pval:.4f}"
    else:
        pval_str = f"p={pval:.3f}"

    return f"{pval_str} {stars}".strip()


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    Compute pairwise correlation matrix with optional filtering.

    Parameters
    ----------
    df : pd.DataFrame
        Columns are variables; rows are observations.
    method : str
        'spearman' (default) or 'pearson'.
    min_samples : int
        Minimum non-null pairs required; otherwise NaN.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix.
    """
    return df.corr(method=method, min_periods=min_samples)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split maintaining class proportions.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
        Binary labels.
    test_size : float
        Fraction of samples for test set.
    random_state : int

    Returns
    -------
    Tuple: X_train, X_test, y_train, y_test
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(sss.split(X, y))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    logger.info(
        "Split: %d train (%.0f%% positive) / %d test (%.0f%% positive)",
        len(X_train), 100 * y_train.mean(),
        len(X_test), 100 * y_test.mean(),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------


def get_timestamp() -> str:
    """Return a compact timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_banner(text: str, width: int = 60) -> None:
    """Print a formatted banner line for pipeline step announcements."""
    border = "=" * width
    padded = text.center(width - 4)
    print(f"\n{border}")
    print(f"  {padded}  ")
    print(f"{border}\n")


def check_sklearn_version() -> str:
    """Return scikit-learn version string."""
    import sklearn
    return sklearn.__version__


def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> str:
    """
    Print a concise summary of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    name : str

    Returns
    -------
    str : Summary text.
    """
    n_rows, n_cols = df.shape
    n_null = df.isna().sum().sum()
    null_pct = 100 * n_null / max(n_rows * n_cols, 1)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    summary = (
        f"\n--- {name} Summary ---\n"
        f"  Shape:        {n_rows} rows x {n_cols} cols\n"
        f"  Null values:  {n_null} ({null_pct:.1f}%)\n"
        f"  Numeric cols: {len(numeric_cols)}\n"
        f"  Columns:      {list(df.columns[:8])}{'...' if n_cols > 8 else ''}\n"
    )
    print(summary)
    return summary
