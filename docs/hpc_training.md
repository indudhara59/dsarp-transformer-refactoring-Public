# HPC training

Replace every `REPLACE_*` Slurm placeholder, create the environment, set `DSARP_DATA_DIR`, `DSARP_OUTPUT_DIR`, and `DSARP_MODEL_CACHE`, then run `sbatch hpc/submit_stage2_pipeline.sh`. It submits dataset → training → evaluation → calibration → embeddings → inference with `afterok` dependencies.

The loop supports AMP, gradient accumulation/clipping, scheduling hooks, early stopping, checkpoint metadata, resume primitives, frozen/unfreezing configuration, DDP launch with `torchrun`, optional FSDP configuration, TensorBoard paths, and optional MLflow. Adapt modules and resource requests to the university cluster; none are guessed here.

