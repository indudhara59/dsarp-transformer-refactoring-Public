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


if __name__ == "__main__":
    app()

