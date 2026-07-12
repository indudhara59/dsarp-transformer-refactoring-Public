# Model evaluation

Macro-F1 is primary. Classification includes macro precision/recall/F1, weighted F1, accuracy, balanced accuracy, per-class precision/recall, confusion matrices, ROC-AUC and PR-AUC when valid. Regression includes MAE, RMSE, Pearson, Spearman, and R². Calibration includes ECE, Brier score, and reliability bins before/after fitting.

Default grouped splits keep every smell instance together. Leave-one-project-out and temporal cross-version manifests are available when data permits. Results should be sliced by smell, recommendation, project, version, confidence, and expert versus weak provenance.

