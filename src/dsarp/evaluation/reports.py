"""Machine- and human-readable evaluation reports."""
import json
from pathlib import Path
from typing import Any


def write_evaluation_report(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"evaluation.json").write_text(json.dumps(metrics,indent=2,sort_keys=True,default=str))
    lines=["# DSARP Stage 2 evaluation",""]
    for section,value in metrics.items(): lines.extend([f"## {section}","",f"```json\n{json.dumps(value,indent=2,default=str)}\n```",""])
    (output_dir/"evaluation.md").write_text("\n".join(lines))
    (output_dir/"evaluation.html").write_text("<!doctype html><meta charset='utf-8'><title>DSARP evaluation</title><h1>DSARP Stage 2 evaluation</h1><pre>"+json.dumps(metrics,indent=2,default=str).replace("&","&amp;").replace("<","&lt;")+"</pre>")

