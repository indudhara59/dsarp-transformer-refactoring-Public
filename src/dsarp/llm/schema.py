"""Strict grounded LLM validation response."""
from pydantic import BaseModel,ConfigDict,Field


class LLMValidationResponse(BaseModel):
    model_config=ConfigDict(extra="forbid")
    valid:bool; recommendation_id:str
    applicability_score:float=Field(ge=0,le=1); benefit_score:float=Field(ge=0,le=1); risk_score:float=Field(ge=0,le=1); semantic_preservation_score:float=Field(ge=0,le=1); evidence_strength:float=Field(ge=0,le=1); confidence_score:float=Field(ge=0,le=1)
    reasoning:list[str]; warnings:list[str]; implementation_steps:list[str]; insufficient_evidence:bool; requested_additional_context:list[str]

