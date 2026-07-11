"""Standalone HTML report rendering."""
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


def write_html(context: dict[str, Any], path: Path, template_dir: Path | None = None) -> None:
    """Render the Stage 1 report with HTML escaping enabled."""
    directory = template_dir or Path(__file__).parent / "templates"
    environment = Environment(loader=FileSystemLoader(directory), autoescape=select_autoescape(["html", "xml"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(environment.get_template("report.html.j2").render(**context), encoding="utf-8")

