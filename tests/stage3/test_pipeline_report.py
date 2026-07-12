from pathlib import Path
import pandas as pd
from dsarp.final_reporting.reports import MAIN_COLUMNS,validate_final_schema
from dsarp.pipeline.full_pipeline import FullPipeline


def test_final_schema():
    validate_final_schema(pd.DataFrame([{column:0 for column in MAIN_COLUMNS}]))


def test_resume_and_dry_run(tmp_path:Path):
    pipeline=FullPipeline({"output_dir":str(tmp_path),"dry_run":False,"force":False}); assert pipeline.stage("x",lambda:3)==3; assert pipeline.completed("x"); called=[]; assert pipeline.stage("x",lambda:called.append(1)) is None; assert not called
    dry=FullPipeline({"output_dir":str(tmp_path/"dry"),"dry_run":True}); assert dry.stage("never",lambda:called.append(2)) is None; assert not (tmp_path/"dry/stage_manifests/never.json").exists()
