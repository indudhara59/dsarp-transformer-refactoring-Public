"""DSARP command-line interface."""
from pathlib import Path

import typer
from rich.console import Console

from dsarp.exceptions import DsarpError
from dsarp.logging_config import configure_logging
from dsarp.pipeline.stage1_pipeline import inspect_data, run_stage1

app = typer.Typer(help="Rank architectural refactoring recommendations from Arcan exports.", no_args_is_help=True)
console = Console()


def _execute(data_dir: Path, output_dir: Path, top_k: int) -> None:
    configure_logging()
    try:
        paths = run_stage1(data_dir, output_dir, top_k)
    except DsarpError as exc:
        console.print(f"[red]Stage 1 failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Stage 1 complete.[/green] Wrote {len(paths)} artifacts below {output_dir}")


@app.command()
def inspect(data_dir: Path = typer.Option(Path("data")), output_dir: Path = typer.Option(Path("outputs"))) -> None:
    """Inspect CSV schemas and join quality."""
    try:
        report = inspect_data(data_dir, output_dir)
    except DsarpError as exc:
        console.print(f"[red]Inspection failed:[/red] {exc}"); raise typer.Exit(1) from exc
    console.print(f"Inspected {len(report['files'])} files; selected {report['filtering']['selected']} smells.")


@app.command()
def prepare(data_dir: Path = typer.Option(Path("data")), output_dir: Path = typer.Option(Path("outputs"))) -> None:
    """Prepare the canonical dataset (also materializes reproducible downstream artifacts)."""
    _execute(data_dir, output_dir, 20)


@app.command()
def baseline(data_dir: Path = typer.Option(Path("data")), output_dir: Path = typer.Option(Path("outputs")), top_k: int = typer.Option(20, min=1)) -> None:
    """Generate and rank baseline candidates."""
    _execute(data_dir, output_dir, top_k)


@app.command("run-stage1")
def run_stage1_command(data_dir: Path = typer.Option(Path("data")), output_dir: Path = typer.Option(Path("outputs")), top_k: int = typer.Option(20, min=1)) -> None:
    """Run complete Stage 1."""
    _execute(data_dir, output_dir, top_k)


@app.command("build-dataset")
def build_dataset_command(config: Path = typer.Option(Path("configs/dataset.yaml")), transformer_config: Path = typer.Option(Path("configs/transformer.yaml"))) -> None:
    """Build Stage 2 examples, annotation template, and split manifest."""
    from dsarp.datasets.dataset_builder import build_dataset
    frame = build_dataset(config, transformer_config); console.print(f"[green]Built[/green] {len(frame)} training examples")


def _run_stage2_script(script: str, arguments: list[str]) -> None:
    import subprocess, sys
    completed = subprocess.run([sys.executable, str(Path("scripts") / script), *arguments], check=False)
    if completed.returncode: raise typer.Exit(completed.returncode)


@app.command("train-transformer")
def train_transformer_command(config: Path = typer.Option(Path("configs/training.yaml"))) -> None:
    """Run configured transformer training (intended for HPC)."""
    _run_stage2_script("train_transformer.py", ["--config", str(config)])


@app.command("evaluate-transformer")
def evaluate_transformer_command(predictions: Path = typer.Option(...)) -> None:
    """Evaluate persisted transformer predictions."""
    _run_stage2_script("evaluate_transformer.py", ["--predictions", str(predictions)])


@app.command("calibrate-transformer")
def calibrate_transformer_command(validation: Path = typer.Option(...)) -> None:
    """Fit a validation-set probability calibrator."""
    _run_stage2_script("calibrate_transformer.py", ["--validation", str(validation)])


@app.command("export-embeddings")
def export_embeddings_command(checkpoint: Path = typer.Option(...), dataset: Path = typer.Option(...)) -> None:
    """Export fused embeddings using a local checkpoint."""
    _run_stage2_script("export_embeddings.py", ["--checkpoint", str(checkpoint), "--dataset", str(dataset)])


@app.command("transformer-inference")
def transformer_inference_command(checkpoint: Path = typer.Option(...), dataset: Path = typer.Option(...)) -> None:
    """Run offline batch inference using a local checkpoint."""
    _run_stage2_script("run_transformer_inference.py", ["--checkpoint", str(checkpoint), "--dataset", str(dataset)])


@app.command("build-rank-dataset")
def build_rank_dataset_command(candidates: Path = typer.Option(Path("outputs/processed/refactoring_candidates.csv")), predictions: Path | None = typer.Option(Path("outputs/transformer_predictions/candidate_predictions.csv"))) -> None:
    """Build all three rank datasets."""
    from dsarp.ranking.dataset_builder import build_rank_dataset
    paths=build_rank_dataset(candidates,predictions); console.print(f"[green]Built[/green] {len(paths)} rank datasets")


@app.command("train-ranker")
def train_ranker_command() -> None: _run_stage2_script("train_ranker.py",[])


@app.command("tune-ranker")
def tune_ranker_command() -> None: _run_stage2_script("tune_ranker.py",[])


@app.command("evaluate-ranker")
def evaluate_ranker_command() -> None: _run_stage2_script("evaluate_ranker.py",[])


@app.command("validate-with-llm")
def validate_with_llm_command() -> None: _run_stage2_script("validate_with_llm.py",[])


@app.command("run")
def run_command(data_dir:Path=typer.Option(Path("data")),output_dir:Path=typer.Option(Path("outputs")),top_k:int=typer.Option(20),transformer_model:Path|None=typer.Option(None),ranker_model:Path|None=typer.Option(None),calibrator:Path|None=typer.Option(None),llm_backend:str=typer.Option("disabled"),llm_model:str|None=typer.Option(None),llm_top_n:int=typer.Option(20),dry_run:bool=typer.Option(False),force:bool=typer.Option(False))->None:
    """Run the complete three-stage system."""
    from dsarp.pipeline.full_pipeline import run_full_pipeline
    paths=run_full_pipeline(overrides={"data_dir":str(data_dir),"output_dir":str(output_dir),"top_k":top_k,"transformer_model":str(transformer_model) if transformer_model else None,"ranker_model":str(ranker_model) if ranker_model else None,"calibrator_dir":str(calibrator) if calibrator else None,"llm_backend":llm_backend,"llm_model":llm_model,"llm_top_n":llm_top_n,"dry_run":dry_run,"force":force}); console.print(f"[green]Final pipeline complete[/green]: {len(paths)} outputs")


@app.command("run-all")
def run_all_command(config:Path=typer.Option(Path("configs/full_pipeline.yaml")))->None:
    from dsarp.pipeline.full_pipeline import run_full_pipeline
    run_full_pipeline(config)


@app.command("ensemble")
def ensemble_command() -> None: console.print("Ensemble scoring is executed by `dsarp run` using configs/ensemble.yaml.")


@app.command("diversify")
def diversify_command() -> None: console.print("Diversity selection is executed by `dsarp run` using configs/ensemble.yaml.")


@app.command("generate-final-report")
def generate_final_report_command() -> None: console.print("Final reports are generated by `dsarp run`; use --force to regenerate.")


@app.command("recommend")
def recommend_command(
    smells: Path = typer.Option(...), affects: Path = typer.Option(...), metrics: Path = typer.Option(...),
    transformer_model: Path | None = typer.Option(None), ranker_model: Path | None = typer.Option(None),
    top_k: int = typer.Option(20, min=1), output_dir: Path = typer.Option(Path("outputs/current-system")),
    baseline_only: bool = typer.Option(False), dry_run: bool = typer.Option(False), force: bool = typer.Option(False),
) -> None:
    """Recommend from the three required Arcan CSV files."""
    from dsarp.pipeline.csv_inference_pipeline import run_csv_inference
    paths = run_csv_inference(smells, affects, metrics, output_dir, top_k, transformer_model, ranker_model, baseline_only, dry_run, force)
    console.print(f"[green]Recommendation pipeline complete[/green]: {len(paths)} outputs")


@app.command("select-commits")
def select_commits_command(
    repository: Path = typer.Option(...), output: Path = typer.Option(Path("outputs/historical/selected_commits.csv")),
    max_commits: int | None = typer.Option(None), sample_size: int | None = typer.Option(None), seed: int = typer.Option(42),
    dry_run: bool = typer.Option(False),
) -> None:
    """Select non-merge Java commits; message keywords only nominate candidates."""
    import pandas as pd
    from dsarp.mining.commit_selector import CommitSelectionConfig, select_commits
    from dsarp.mining.repository_manager import RepositoryManager
    if dry_run: console.print(f"Would inspect {repository}"); return
    rows = select_commits(RepositoryManager(repository), CommitSelectionConfig(max_commits=max_commits, sample_size=sample_size, seed=seed))
    output.parent.mkdir(parents=True, exist_ok=True); frame = pd.DataFrame(rows); frame.insert(0, "repository", repository.name); frame.to_csv(output, index=False)
    console.print(f"[green]Selected[/green] {len(frame)} commits into {output}")


@app.command("build-historical-dataset")
def build_historical_dataset_command(
    data_dir: Path = typer.Option(Path("data/historical")), output_dir: Path = typer.Option(Path("outputs/historical")),
    resume: bool = typer.Option(True), dry_run: bool = typer.Option(False), force: bool = typer.Option(False),
) -> None:
    """Build candidate-level historical labels from prepared before/after snapshots."""
    from dsarp.pipeline.historical_training_data_pipeline import HistoricalPipeline
    path = HistoricalPipeline(output_dir, resume=resume, dry_run=dry_run, force=force).run_prepared(data_dir)
    console.print(f"[green]Historical dataset[/green]: {path}")


@app.command("build-dataset-variants")
def build_dataset_variants_command(dataset: Path = typer.Option(...), output_dir: Path = typer.Option(Path("outputs/historical/datasets/variants")), top_k: int = typer.Option(3), confidence_threshold: float = typer.Option(.35)) -> None:
    """Materialize research datasets A-F (E is recommended)."""
    import pandas as pd
    from dsarp.historical.dataset_builder import build_variants
    paths = build_variants(pd.read_csv(dataset), output_dir, top_k, confidence_threshold); console.print(f"[green]Built[/green] {len(paths)} variants")


@app.command("inspect-historical-dataset")
def inspect_historical_dataset_command(dataset: Path = typer.Option(...)) -> None:
    """Print a lightweight historical dataset quality summary."""
    import pandas as pd
    frame = pd.read_csv(dataset); console.print({"rows": len(frame), "repositories": frame["repository"].nunique(), "commits": frame["commit"].nunique(), "smells": frame["smell_type"].value_counts().to_dict(), "labels": frame["relevance_grade"].value_counts().to_dict()})


@app.command("run-integrated-training-data-pipeline")
def run_integrated_training_data_pipeline_command(
    repositories_config: Path = typer.Option(Path("configs/repositories.yaml")), output_dir: Path = typer.Option(Path("outputs/historical")),
    resume: bool = typer.Option(True), dry_run: bool = typer.Option(False), force: bool = typer.Option(False),
) -> None:
    """Run safe local orchestration over prepared snapshots; external tools remain config-driven."""
    from dsarp.pipeline.historical_training_data_pipeline import HistoricalPipeline
    _ = repositories_config
    path = HistoricalPipeline(output_dir, resume=resume, dry_run=dry_run, force=force).run_prepared(Path("data/historical")); console.print(path)


@app.command("mine-refactorings")
def mine_refactorings_command() -> None:
    """Describe the configured RefactoringMiner batch entry point."""
    console.print("Use scripts/mine_refactorings.py with configs/refactoring_miner.yaml; add --dry-run to validate without execution.")


@app.command("run-arcan-history")
def run_arcan_history_command() -> None:
    """Describe the configured before/after Arcan batch entry point."""
    console.print("Use scripts/run_arcan_snapshots.py with configs/arcan_runner.yaml; add --dry-run to validate without execution.")


@app.command("match-smells")
def match_smells_command() -> None:
    """Match prepared snapshots as part of historical dataset construction."""
    console.print("Smell matching is executed by `dsarp build-historical-dataset`.")


@app.command("build-historical-rank-data")
def build_historical_rank_data_command() -> None:
    """Build rank data through the existing rank-dataset command after Dataset E is split."""
    console.print("Run scripts/build_historical_rank_dataset.py on the training-only Dataset E split.")


@app.command("analyze-repository")
def analyze_repository_command(
    repository: Path = typer.Option(...), version: str = typer.Option("HEAD"),
    arcan_config: Path = typer.Option(Path("configs/arcan_runner.yaml")), top_k: int = typer.Option(20),
    output_dir: Path = typer.Option(Path("outputs/project-version")), transformer_model: Path | None = typer.Option(None),
    ranker_model: Path | None = typer.Option(None), baseline_only: bool = typer.Option(False), dry_run: bool = typer.Option(False), force: bool = typer.Option(False),
) -> None:
    """Analyze one local repository revision, then call the shared CSV recommender."""
    import os
    import yaml
    from dsarp.adapters.repository_inference import analyze_repository
    from dsarp.arcan.runner import SubprocessArcanRunner
    config = yaml.safe_load(arcan_config.read_text())
    command = [os.path.expandvars(str(part)) for part in config["command"]]
    if dry_run: console.print({"repository": str(repository), "version": version, "arcan_command": command}); return
    runner = SubprocessArcanRunner(command, int(config["timeout_seconds"]), str(config.get("version", "unknown")))
    paths = analyze_repository(repository, version, output_dir, runner, top_k=top_k, transformer_model=transformer_model, ranker_model=ranker_model, baseline_only=baseline_only, force=force)
    console.print(f"[green]Repository recommendation complete[/green]: {len(paths)} outputs")


if __name__ == "__main__":
    app()
