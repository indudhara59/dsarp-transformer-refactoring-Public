"""Persist prediction tables and embedding indexes."""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsarp.inference.predictor import predict_batch


def run_batch_inference(model:Any,loader:Any,output_dir:Path,embedding_dir:Path,model_version:str,calibrator:Any|None=None)->pd.DataFrame:
    records=[]; vectors=[]
    for batch in loader:
        predictions,embeddings=predict_batch(model,batch,model_version,calibrator); records.extend(predictions); vectors.append(embeddings)
    frame=pd.DataFrame(records); output_dir.mkdir(parents=True,exist_ok=True); embedding_dir.mkdir(parents=True,exist_ok=True); frame.to_csv(output_dir/"candidate_predictions.csv",index=False)
    try: frame.to_parquet(output_dir/"candidate_predictions.parquet",index=False)
    except (ImportError,ValueError): pass
    matrix=np.concatenate(vectors) if vectors else np.empty((0,0)); np.save(embedding_dir/"candidate_embeddings.npy",matrix); pd.DataFrame({"candidate_id":frame.get("candidate_id",[]),"embedding_row":range(len(frame))}).to_csv(embedding_dir/"candidate_embedding_index.csv",index=False)
    (output_dir/"inference_metadata.json").write_text(json.dumps({"model_version":model_version,"rows":len(frame)},indent=2)); return frame

