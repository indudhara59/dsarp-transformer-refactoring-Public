"""Nine required research experiment definitions and tables."""
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPERIMENTS={"A_severity_only":{"features":"severity"},"B_atdi_only":{"features":"atdi"},"C_stage1_rules":{"features":"stage1"},"D_numeric_pointwise":{"model":"pointwise"},"E_transformer_only":{"features":"transformer"},"F_lambdamart_no_transformer":{"model":"lambdamart","transformer":False},"G_lambdamart_transformer":{"model":"lambdamart","transformer":True},"H_lambdamart_transformer_llm":{"model":"lambdamart","transformer":True,"llm":True},"I_full_diversity":{"model":"full","diversity":True}}


def write_ablation_results(results:list[dict[str,Any]],output_dir:Path)->None:
    output_dir.mkdir(parents=True,exist_ok=True); frame=pd.DataFrame(results); frame.to_csv(output_dir/"ablation_results.csv",index=False); (output_dir/"ablation_results.json").write_text(json.dumps(results,indent=2,default=str)); (output_dir/"ablation_table.md").write_text(frame.to_markdown(index=False) if not frame.empty else "# Ablation results\n\nNot executed.")

