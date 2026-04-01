"""
DDR Biomarker & Drug Response Prediction Pipeline.

Sub-modules:
    data_loader         -- GDSC2 and DepMap data loading & merging
    feature_engineering -- Feature construction (mutations, HRD score, MSI)
    models              -- Model training, cross-validation, serialization
    evaluation          -- Metrics, ROC/PR plots, classification reports
    biomarker_analysis  -- SHAP explainability, Mann-Whitney U, heatmaps
    utils               -- Logging, I/O, statistical helpers
"""

__version__ = "1.0.0"
__author__ = "DDR Biomarker Pipeline"
