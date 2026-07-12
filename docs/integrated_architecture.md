# Integrated architecture

Historical training analyzes parent/current snapshots with Arcan, associates RefactoringMiner operations to individual smell areas, and creates confidence-weighted candidate labels. Manual inference keeps the three Arcan CSV files as its complete required input and uses only current-state features. Optional repository inference checks out one version, runs Arcan once, then calls that same CSV pipeline.

Arcan is authoritative for `godComponent`, `unstableDep`, and `hubLikeDep` in both workflows. Custom detectors are not final ground truth. RefactoringMiner detects concrete operations; high-level strategies are explicitly inferred by configured rules.
