"""
Layer 1: PII Detection using Microsoft Presidio

This module handles entity detection using pattern matching and NER.
Detects: Names, Emails, Phone Numbers, Credit Cards, SSNs, etc.
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Entity:
    """Detected PII entity"""
    type: str
    value: str
    start: int
    end: int
    score: float = 0.95


class PIIDetector:
    """
    PII entity detector using pattern matching
    
    In production, replace with:
        from presidio_analyzer import AnalyzerEngine
    """
    
    def __init__(self):
        # Pattern definitions for common PII types
        self.patterns = {
            'PERSON': r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b',
            'EMAIL_ADDRESS': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE_NUMBER': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b',
            'CREDIT_CARD': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'IP_ADDRESS': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'US_DRIVER_LICENSE': r'\b[A-Z]{1,2}\d{5,8}\b',
            'DATE': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        }
    
    def detect(self, text: str) -> List[Entity]:
        """
        Detect all PII entities in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of Entity objects with detected PII
        """
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            
            for match in matches:
                entities.append(Entity(
                    type=entity_type,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    score=0.95
                ))
        
        # Remove duplicates (overlapping matches)
        entities = self._remove_overlaps(entities)
        
        return entities
    
    def _remove_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entity detections"""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda e: (e.start, -len(e.value)))
        
        # Keep non-overlapping entities
        result = [entities[0]]
        
        for entity in entities[1:]:
            if entity.start >= result[-1].end:
                result.append(entity)
        
        return result
    
    def get_supported_entities(self) -> List[str]:
        """Return list of supported entity types"""
        return list(self.patterns.keys())


class PresidioPIIDetector:
    """
    Production version using actual Presidio library
    
    Usage:
        detector = PresidioPIIDetector()
        entities = detector.detect("My email is john@example.com")
    """
    
    def __init__(self):
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore
            from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore
            
            # Configure Presidio with spaCy
            config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}]
            }
            
            provider = NlpEngineProvider(nlp_configuration=config)
            nlp_engine = provider.create_engine()
            
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.using_presidio = True
            
        except ImportError as e:
            print(f"⚠ Presidio not installed ({e}). Using pattern-based fallback.")
            print("  To install Presidio: pip install presidio-analyzer presidio-anonymizer")
            self.fallback_detector = PIIDetector()
            self.using_presidio = False
    
    def detect(self, text: str, language: str = "en") -> List[Entity]:
        """
        Detect PII using Presidio or fallback to patterns
        
        Args:
            text: Input text
            language: Language code (default: "en")
            
        Returns:
            List of Entity objects
        """
        if self.using_presidio:
            results = self.analyzer.analyze(text=text, language=language)
            
            # Convert Presidio results to Entity objects
            return [
                Entity(
                    type=result.entity_type,
                    value=text[result.start:result.end],
                    start=result.start,
                    end=result.end,
                    score=result.score
                )
                for result in results
            ]
        else:
            return self.fallback_detector.detect(text)


# Example usage
if __name__ == "__main__":
    # Test with pattern-based detector
    detector = PIIDetector()
    
    test_texts = [
        "My name is John Smith and my email is john@example.com",
        "Call me at +1-555-123-4567",
        "SSN: 123-45-6789"
    ]
    
    print("=" * 80)
    print("PII DETECTION DEMO")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\nText: {text}")
        entities = detector.detect(text)
        
        if entities:
            for entity in entities:
                print(f"  • {entity.type}: '{entity.value}' (confidence: {entity.score:.2f})")
        else:
            print("  No PII detected")
