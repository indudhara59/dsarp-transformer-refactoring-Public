"""Project/version Markdown and HTML case-study generator."""
import html
from pathlib import Path
import pandas as pd


def generate_case_study(frame:pd.DataFrame,project:str,version_id:str,output_dir:Path)->tuple[Path,Path]:
    subset=frame[(frame.project==project)&(frame.version_id==version_id)].sort_values("final_score",ascending=False).head(10); output_dir.mkdir(parents=True,exist_ok=True); lines=[f"# DSARP case study: {project} / {version_id}","","Top ten recommendations with Arcan, ranker, transformer, LLM, benefit/risk, alternatives, warnings, and context availability.",""]
    for _,r in subset.iterrows(): lines.extend([f"## #{r.get('rank','?')} {r.display_name}",f"Arcan: severity={r.get('severity')}, ATDI={r.get('atdi')}. Ranker={r.get('ranker_score')}, final={r.get('final_score')}. Benefit={r.get('predicted_benefit')}, risk={r.get('predicted_risk')}.",f"Explanation: {r.get('reasoning',[])}",f"Warnings: {r.get('warnings',[])}",f"Source context: {r.get('source_context_available',False)}",""])
    markdown=output_dir/f"{project}_{version_id}.md"; web=output_dir/f"{project}_{version_id}.html"; markdown.write_text("\n".join(lines)); web.write_text("<!doctype html><meta charset='utf-8'><pre>"+html.escape(markdown.read_text())+"</pre>"); return markdown,web

