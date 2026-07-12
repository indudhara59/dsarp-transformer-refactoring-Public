"""Extract and conservatively repair JSON responses."""
import json,re
from typing import Any

from dsarp.llm.schema import LLMValidationResponse


def extract_json(text:str)->dict[str,Any]:
    """Parse a JSON object from plain or fenced output; repair trailing commas only."""
    cleaned=re.sub(r"^```(?:json)?\s*|\s*```$","",text.strip(),flags=re.I)
    start,end=cleaned.find("{"),cleaned.rfind("}")
    if start<0 or end<start: raise ValueError("No JSON object found")
    fragment=re.sub(r",\s*([}\]])",r"\1",cleaned[start:end+1])
    value=json.loads(fragment)
    if not isinstance(value,dict): raise ValueError("LLM response is not an object")
    return value


def validate_response(text:str,expected_recommendation_id:str)->LLMValidationResponse:
    response=LLMValidationResponse.model_validate(extract_json(text))
    if response.recommendation_id!=expected_recommendation_id: raise ValueError("LLM returned a different recommendation_id")
    return response

