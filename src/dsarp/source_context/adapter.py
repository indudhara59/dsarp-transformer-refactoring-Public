"""Optional repository-mining integration without making source mandatory."""
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel,ConfigDict


class ProjectMetadata(BaseModel):
    model_config=ConfigDict(extra="allow")
    repository_url:str|None=None; commit_hash:str|None=None; version:str|None=None; checkout_path:str|None=None; language:str|None=None; build_command:str|None=None; test_command:str|None=None


class SourceContextAdapter:
    def __init__(self,root:Path): self.root=root; self.metadata=self._metadata()
    def _metadata(self)->ProjectMetadata|None:
        path=self.root/"project_metadata.json"; return ProjectMetadata.model_validate(json.loads(path.read_text())) if path.exists() else None
    def component_context(self,component_id:str,component_name:str|None=None)->dict[str,Any]:
        context:dict[str,Any]={"available":False,"basis":"architecture_metrics_only"}
        for filename,key in [("component_classes.csv","classes"),("dependency_edges.csv","dependencies"),("change_history.csv","changes")]:
            path=self.root/filename
            if path.exists():
                frame=pd.read_csv(path); mask=pd.Series(False,index=frame.index)
                for column in ["component_id","affected_component_id","component"]:
                    if column in frame: mask|=frame[column].astype(str).eq(str(component_id))
                if component_name:
                    for column in ["component_name","name"]:
                        if column in frame: mask|=frame[column].astype(str).eq(component_name)
                context[key]=frame[mask].head(100).to_dict("records"); context["available"]=True
        snippets=self.root/"source_context.jsonl"
        if snippets.exists():
            records=[json.loads(line) for line in snippets.read_text().splitlines() if line.strip()]; context["source_snippets"]=[r for r in records if str(r.get("component_id"))==str(component_id)][:20]; context["available"]|=bool(context["source_snippets"])
        if context["available"]: context["basis"]="architecture_metrics_plus_optional_source_context"
        return context

