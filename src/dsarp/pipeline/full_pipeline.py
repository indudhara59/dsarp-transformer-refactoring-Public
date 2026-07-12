"""Resumable failure-bounded DSARP end-to-end pipeline."""
from __future__ import annotations

import hashlib,json,logging
from datetime import UTC,datetime
from pathlib import Path
from typing import Any,Callable

import pandas as pd
import yaml

from dsarp.diversity.mmr import diversify
from dsarp.ensemble.scoring import apply_ensemble
from dsarp.final_reporting.reports import write_final_reports
from dsarp.llm.backends import create_backend
from dsarp.llm.validator import validate_candidates
from dsarp.pipeline.stage1_pipeline import run_stage1

LOGGER=logging.getLogger(__name__)


class FullPipeline:
    """Execute stages with individual manifests and partial-output preservation."""
    def __init__(self,config:dict[str,Any]): self.config=config; self.output=Path(config["output_dir"]); self.manifests=self.output/"stage_manifests"; self.manifests.mkdir(parents=True,exist_ok=True); self.dry_run=bool(config.get("dry_run",False)); self.force=bool(config.get("force",False))
    def _manifest(self,name:str)->Path: return self.manifests/f"{name}.json"
    def completed(self,name:str)->bool: return self._manifest(name).exists() and not self.force
    def stage(self,name:str,action:Callable[[],Any])->Any:
        if self.completed(name): LOGGER.info("Skipping completed stage %s",name); return None
        if self.dry_run: LOGGER.info("Dry run: would execute %s",name); return None
        started=datetime.now(UTC).isoformat()
        try: result=action(); status="completed"
        except Exception as exc:
            self._manifest(name).write_text(json.dumps({"stage":name,"status":"failed","started":started,"error":str(exc)},indent=2)); LOGGER.exception("Stage failed: %s",name); raise
        self._manifest(name).write_text(json.dumps({"stage":name,"status":status,"started":started,"finished":datetime.now(UTC).isoformat()},indent=2)); return result
    def run(self)->dict[str,Path]:
        data_dir=Path(self.config["data_dir"]); top_k=int(self.config["top_k"])
        self.stage("stage1",lambda:run_stage1(data_dir,self.output,top_k))
        candidates_path=self.output/"processed/refactoring_candidates.csv"
        if self.dry_run: return {}
        def load_candidates()->pd.DataFrame:
            frame=pd.read_csv(candidates_path); predictions=Path(self.config.get("transformer_predictions",""))
            if predictions.exists(): frame=frame.merge(pd.read_csv(predictions),on="candidate_id",how="left",validate="one_to_one")
            return frame
        frame=load_candidates()
        # Loaded ranker predictions take precedence; a transparent Stage 1 fallback keeps
        # the three-CSV core operational when trained artifacts are unavailable.
        ranker_path=Path(self.config.get("ranker_model",""))
        if ranker_path.exists():
            def ranker_inference()->pd.Series:
                import xgboost as xgb
                model=xgb.Booster(); model.load_model(ranker_path); manifest=ranker_path.with_suffix(".features.json"); features=json.loads(manifest.read_text()); matrix=frame.reindex(columns=features).apply(pd.to_numeric,errors="coerce").fillna(0); return pd.Series(model.predict(xgb.DMatrix(matrix,feature_names=features)),index=frame.index)
            frame["ranker_score"]=self.stage("ranker_inference",ranker_inference) if not self.completed("ranker_inference") else ranker_inference()
        if "ranker_score" not in frame: frame["ranker_score"]=pd.to_numeric(frame.get("final_priority_score",0),errors="coerce").fillna(0)
        llm_config=yaml.safe_load(Path("configs/llm.yaml").read_text()); llm_config["backend"]=self.config.get("llm_backend",llm_config["backend"]); llm_config["top_n"]=self.config.get("llm_top_n",llm_config["top_n"])
        validations,errors=self.stage("llm_validation",lambda:validate_candidates(frame,create_backend(llm_config),llm_config)) or (pd.DataFrame(),pd.DataFrame())
        if not validations.empty: frame=frame.merge(validations,on="candidate_id",how="left",suffixes=("","_llm"),validate="one_to_one")
        ensemble_config=yaml.safe_load(Path("configs/ensemble.yaml").read_text()); raw,rejected=self.stage("ensemble",lambda:apply_ensemble(frame,ensemble_config)) or apply_ensemble(frame,ensemble_config)
        diversified=self.stage("diversity",lambda:diversify(raw,top_k,ensemble_config["diversity"])) or diversify(raw,top_k,ensemble_config["diversity"])
        metadata={"run_id":hashlib.sha256(f"{data_dir}|{top_k}".encode()).hexdigest()[:16],"created_at":datetime.now(UTC).isoformat(),"config":self.config,"models":{"transformer":self.config.get("transformer_model"),"ranker":self.config.get("ranker_model"),"calibrator":self.config.get("calibrator_dir")},"llm_errors":len(errors),"source_context":"architecture_metrics_only unless optional adapter attached"}
        return self.stage("final_report",lambda:write_final_reports(raw,diversified,rejected,self.output/"final",metadata)) or {}


def run_full_pipeline(config_path:Path=Path("configs/full_pipeline.yaml"),overrides:dict[str,Any]|None=None)->dict[str,Path]:
    config=yaml.safe_load(config_path.read_text()); config.update({k:v for k,v in (overrides or {}).items() if v is not None}); return FullPipeline(config).run()
