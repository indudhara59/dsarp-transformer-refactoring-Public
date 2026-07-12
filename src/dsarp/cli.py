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


if __name__ == "__main__":
    app()
