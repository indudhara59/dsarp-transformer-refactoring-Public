"""Grounded Stage 3 validation prompt."""
import json
from typing import Any

PROMPT_VERSION="dsarp-validation-v1"
EVIDENCE_FIELDS=["project","version_id","smell_type","smell_id","affected_component_name","central_component","affected_elements","severity","strength","atdi","atdi_weighted","smell_size","number_of_edges","fan_in","fan_out","instability_metric","lines_of_code","number_of_children","page_rank","rule_evidence","applicability_probability","predicted_relevance_grade","predicted_benefit","predicted_risk","ranker_score","recommendation_id","display_name","taxonomy_description","intended_effect","contraindications","source_context"]


def build_validation_prompt(candidate:dict[str,Any],prompt_version:str=PROMPT_VERSION)->str:
    evidence={key:candidate.get(key,"<MISSING>") for key in EVIDENCE_FIELDS}
    return f"""DSARP ARCHITECTURAL REFACTORING VALIDATION ({prompt_version})
Arcan has already detected the architectural smell. Do not redetect it. Validate only the proposed recommendation.
Use only supplied evidence. Do not invent components, classes, packages, history, compilation, tests, or measured improvements. Acknowledge uncertainty and request context when needed.
Return one strict JSON object matching these exact fields: valid (boolean), recommendation_id (string), applicability_score, benefit_score, risk_score, semantic_preservation_score, evidence_strength, confidence_score (all 0..1), reasoning (string array), warnings (string array), implementation_steps (string array), insufficient_evidence (boolean), requested_additional_context (string array).
EVIDENCE_JSON:
{json.dumps(evidence,sort_keys=True,default=str)}"""

