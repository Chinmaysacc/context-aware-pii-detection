from pydantic import BaseModel
from typing import List


class AnalyzeRequest(BaseModel):
    text: str


class EntityResult(BaseModel):
    entity_type: str
    entity_value: str
    is_pii: bool
    context_label: str
    context_confidence: float
    reasoning: str


class AnalyzeResponse(BaseModel):
    input_text: str
    entities_found: int
    pii_detected: int
    results: List[EntityResult]
