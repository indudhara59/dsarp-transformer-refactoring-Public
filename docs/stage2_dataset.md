# Stage 2 dataset

Each row is one smell instance × affected component × proposed recommendation. Identifiers, deterministic semantic context, the configured numeric vector, categorical values, five Stage 1 baseline scores, and nullable expert/weak labels are retained. `configs/dataset.yaml` controls columns and splits. `build-dataset` writes Parquet when PyArrow is available, otherwise CSV, plus `split_manifest.csv`.

Weak supervision is marked `label_source=weak_supervision`; it is not ground truth. Synthetic easy, hard, and counterfactual negatives receive new stable IDs and never overwrite positives.

