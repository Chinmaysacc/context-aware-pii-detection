from fastapi import FastAPI
from app.schemas import AnalyzeRequest, AnalyzeResponse
from src.pipeline import ContextAwarePIISystem

app = FastAPI(
    title="Context-Aware PII Detection API",
    description="Detects PII using NER + Context + GenAI reasoning",
    version="1.0"
)

# Initialize system once (important)
pii_system = ContextAwarePIISystem(
    use_ml_classifier=False,   # later switch to True
    use_genai=False            # later switch to True
)


@app.get("/")
def health_check():
    return {"status": "running"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest):
    result = pii_system.analyze(request.text)

    return {
        "input_text": result["input_text"],
        "entities_found": result["entities_found"],
        "pii_detected": result["pii_detected"],
        "results": [
            {
                "entity_type": r["entity_type"],
                "entity_value": r["entity_value"],
                "is_pii": r["is_pii"],
                "context_label": r["context_label"],
                "context_confidence": r["context_confidence"],
                "reasoning": r["reasoning"]
            }
            for r in result["results"]
        ]
    }
