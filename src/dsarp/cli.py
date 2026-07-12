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


if __name__ == "__main__":
    app()
