# Learning to rank

DSARP trains separate rank datasets for smell prioritization (`project, version`), candidate selection (`project, version, smell, component`), and global ranking (`project, version`). Groups are sorted contiguously into qids and never split between partitions. Label confidence becomes sample weight; expert and historical labels are primary, while weak labels remain explicitly auxiliary.

XGBoost `rank:ndcg` is primary with NDCG@5/10 and MAP@10. CPU `hist` is the safe default; HPC can set CUDA explicitly. Pointwise regression, severity, ATDI, Stage 1, transformer, optional LightGBM, and research neural baselines are nonmandatory alternatives. PCA/SVD reducers must be fit on training embeddings only.

