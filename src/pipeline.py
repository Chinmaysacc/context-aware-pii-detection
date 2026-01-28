"""
Complete PII Detection Pipeline

Integrates all three layers:
1. PII Detection (NER / Regex / Presidio)
2. Context Classification (Rule-based / ML)
3. Reasoning Generation (GenAI / Templates)

Uses ENTITY-CONTEXT WINDOW for correct mixed-intent handling
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from src.pii_detector import PIIDetector, Entity
from src.context_classifier import ContextClassifier, ContextResult
from src.reasoning_generator import ReasoningGenerator


@dataclass
class PIIAnalysisResult:
    entity_type: str
    entity_value: str
    entity_start: int
    entity_end: int
    detection_score: float
    context_label: str
    context_confidence: float
    is_pii: bool
    reasoning: str


class ContextAwarePIISystem:
    """
    Context-Aware PII Detection System
    """

    DECISION_RULES = {
        "personal_data": True,
        "example_demo": False,
        "educational": False,
        "public_info": False,
        "unknown": True  # privacy-first default
    }

    def __init__(
        self,
        use_ml_classifier: bool = False,
        use_genai: bool = False,
        api_key: Optional[str] = None
    ):
        # Layer 1: PII detector
        self.pii_detector = PIIDetector()

        # Layer 2: Context classifier
        if use_ml_classifier:
            from src.context_classifier import MLContextClassifier
            self.context_classifier = MLContextClassifier()
            print("ℹ Using ML context classifier")
        else:
            self.context_classifier = ContextClassifier()
            print("✓ Using rule-based context classifier")

        # Layer 3: Reasoning
        self.reasoning_generator = ReasoningGenerator(
            api_key=api_key,
            use_api=use_genai
        )

    # ------------------------------------------------------------------
    # Helper: Entity-level context window (SAFE & ROBUST)
    # ------------------------------------------------------------------

    def _get_entity_context_window(
        self,
        text: str,
        entity: Entity,
        window_size: int = 50
    ) -> str:
        """
        Extract character-based context window around entity.
        Robust to punctuation, emails, numbers, etc.
        """

        start = max(0, entity.start - window_size)
        end = min(len(text), entity.end + window_size)

        return text[start:end]

    def _decide_pii(self, context: ContextResult) -> bool:
        """Decide PII based on context"""
        return self.DECISION_RULES.get(context.label, True)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Dict:
        """
        Perform full PII analysis
        """

        # Step 1: Detect candidate entities
        entities = self.pii_detector.detect(text)

        if not entities:
            return {
                "input_text": text,
                "entities_found": 0,
                "pii_detected": 0,
                "results": []
            }

        results = []

        # Step 2: Entity-wise context classification
        for entity in entities:
            context_window = self._get_entity_context_window(text, entity)

            context_result = self.context_classifier.classify(context_window)

            is_pii = self._decide_pii(context_result)

            reasoning = self.reasoning_generator.generate(
                entity_type=entity.type,
                entity_value=entity.value,
                context_label=context_result.label,
                context_confidence=context_result.confidence,
                is_pii=is_pii,
                original_text=context_window
            )

            results.append(
                PIIAnalysisResult(
                    entity_type=entity.type,
                    entity_value=entity.value,
                    entity_start=entity.start,
                    entity_end=entity.end,
                    detection_score=entity.score,
                    context_label=context_result.label,
                    context_confidence=context_result.confidence,
                    is_pii=is_pii,
                    reasoning=reasoning
                )
            )

        return {
            "input_text": text,
            "entities_found": len(results),
            "pii_detected": sum(1 for r in results if r.is_pii),
            "results": [asdict(r) for r in results]
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        return [self.analyze(text) for text in texts]
