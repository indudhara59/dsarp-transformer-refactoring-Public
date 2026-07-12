"""Model prediction and schema normalization."""
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch


def predict_batch(model: Any,batch: dict[str,Any],model_version:str,calibrator:Any|None=None)->tuple[list[dict[str,Any]],np.ndarray]:
    model.eval()
    with torch.no_grad(): output=model(**{k:v for k,v in batch.items() if k not in {"candidate_id","labels"}})
    probability=torch.sigmoid(output.applicability_logits).squeeze(-1).cpu().numpy(); relevance=torch.softmax(output.relevance_logits,dim=-1).cpu().numpy()
    if calibrator is not None: probability=calibrator.predict_proba(probability)
    benefit=output.predicted_benefit.squeeze(-1).cpu().numpy(); risk=output.predicted_risk.squeeze(-1).cpu().numpy(); embeddings=output.fused_embedding.cpu().numpy(); uncertainty=output.uncertainty.squeeze(-1).cpu().numpy() if output.uncertainty is not None else 1-np.abs(probability-.5)*2
    metadata={"timestamp":datetime.now(UTC).isoformat(),"device":str(next(model.parameters()).device)}
    records=[{"candidate_id":str(candidate),"applicability_probability":float(probability[i]),"predicted_relevance_probabilities":relevance[i].tolist(),"predicted_relevance_grade":int(relevance[i].argmax()),"predicted_benefit":float(benefit[i]),"predicted_risk":float(risk[i]),"calibrated_confidence":float(1-uncertainty[i]),"fused_embedding_reference":f"candidate_embeddings.npy:{i}","model_version":model_version,"inference_metadata":metadata} for i,candidate in enumerate(batch["candidate_id"])]
    return records,embeddings

