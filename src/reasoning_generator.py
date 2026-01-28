"""
Layer 3: Reasoning Generation using GenAI

This module generates human-readable explanations for PII decisions
using Claude API or template-based fallback.
"""

import os
from typing import Optional


class ReasoningGenerator:
    """
    GenAI-powered reasoning generator
    
    Generates natural language explanations for why something
    is or isn't classified as PII.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_api: bool = True):
        """
        Initialize reasoning generator
        
        Args:
            api_key: Anthropic API key (optional)
            use_api: Whether to use API or template fallback
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.use_api = use_api and self.api_key is not None
        
        if self.use_api:
            try:
                import anthropic  # type: ignore
                self.client = anthropic.Anthropic(api_key=self.api_key)
                print("✓ GenAI API initialized")
            except ImportError as e:
                print(f"⚠ anthropic library not installed ({e}). Using template fallback.")
                print("  To install: pip install anthropic")
                self.use_api = False
        else:
            print("ℹ Using template-based reasoning (no API)")
    
    def generate(
        self,
        entity_type: str,
        entity_value: str,
        context_label: str,
        context_confidence: float,
        is_pii: bool,
        original_text: str
    ) -> str:
        """
        Generate reasoning explanation
        
        Args:
            entity_type: Type of entity detected
            entity_value: Value of the entity
            context_label: Classified context
            context_confidence: Confidence score
            is_pii: Whether classified as PII
            original_text: Original input text
            
        Returns:
            Natural language explanation
        """
        if self.use_api:
            return self._generate_with_api(
                entity_type, entity_value, context_label,
                context_confidence, is_pii, original_text
            )
        else:
            return self._generate_with_template(
                entity_type, entity_value, context_label,
                context_confidence, is_pii
            )
    
    def _generate_with_api(
        self,
        entity_type: str,
        entity_value: str,
        context_label: str,
        confidence: float,
        is_pii: bool,
        original_text: str
    ) -> str:
        """Generate reasoning using Claude API"""
        
        prompt = f"""You are a PII (Personally Identifiable Information) detection expert analyzing results from an AI system.

Detection Details:
- Entity Type: {entity_type}
- Detected Value: "{entity_value}"
- Text Context Classification: {context_label}
- Classification Confidence: {confidence:.2%}
- PII Decision: {"PII" if is_pii else "NOT PII"}
- Original Text: "{original_text}"

Task: Explain in 2-3 clear, professional sentences WHY this decision was made. Your explanation should:
1. Reference the specific entity and context
2. Explain the reasoning behind the classification
3. Use technical but accessible language suitable for an AI/ML academic project

Provide only the explanation, no preamble."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            print(f"API Error: {e}")
            return self._generate_with_template(
                entity_type, entity_value, context_label,
                confidence, is_pii
            )
    
    def _generate_with_template(
        self,
        entity_type: str,
        entity_value: str,
        context_label: str,
        confidence: float,
        is_pii: bool
    ) -> str:
        """Generate reasoning using predefined templates"""
        
        # Format entity type for display
        entity_display = entity_type.replace('_', ' ').lower()
        
        if is_pii:
            templates = {
                'personal_data': (
                    f"The {entity_display} '{entity_value}' appears in a personal context "
                    f"with {confidence:.0%} confidence. The text contains self-referential "
                    f"language indicating this information belongs to a real individual and "
                    f"should be classified as personally identifiable information that requires "
                    f"protection under privacy regulations."
                ),
                'unknown': (
                    f"The system detected a {entity_display} pattern ('{entity_value}') without "
                    f"sufficient contextual markers to determine its purpose. Following privacy-first "
                    f"principles and to err on the side of caution, this is classified as PII to "
                    f"ensure data protection until proven otherwise."
                )
            }
            
        else:
            templates = {
                'example_demo': (
                    f"Although '{entity_value}' matches the pattern of a {entity_display}, "
                    f"the surrounding context clearly indicates this is used as an example or "
                    f"demonstration (confidence: {confidence:.0%}). The text contains explicit "
                    f"markers like 'example', 'for instance', or similar phrases, meaning this "
                    f"is illustrative data rather than real personal information."
                ),
                'educational': (
                    f"The {entity_display} pattern '{entity_value}' appears in an educational "
                    f"or instructional context with {confidence:.0%} confidence. The text is "
                    f"explaining format, structure, or patterns rather than providing actual "
                    f"personal data. This is template information used for documentation or "
                    f"teaching purposes, not real PII."
                ),
                'public_info': (
                    f"The detected {entity_display} '{entity_value}' appears in a public or "
                    f"organizational context. This information is publicly available and "
                    f"represents corporate or institutional contact details rather than "
                    f"an individual's private personal information."
                )
            }
        
        # Get appropriate template
        reasoning = templates.get(
            context_label,
            f"Based on context analysis (confidence: {confidence:.0%}), this "
            f"{entity_display} is classified as {'PII' if is_pii else 'NOT PII'}."
        )
        
        return reasoning


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("REASONING GENERATION DEMO")
    print("=" * 80)
    
    # Initialize generator (will use template fallback if no API key)
    generator = ReasoningGenerator(use_api=False)
    
    # Test cases
    test_cases = [
        {
            'entity_type': 'PHONE_NUMBER',
            'entity_value': '555-1234',
            'context_label': 'personal_data',
            'confidence': 0.85,
            'is_pii': True,
            'text': 'My phone number is 555-1234'
        },
        {
            'entity_type': 'EMAIL_ADDRESS',
            'entity_value': 'user@example.com',
            'context_label': 'example_demo',
            'confidence': 0.90,
            'is_pii': False,
            'text': 'Example email: user@example.com'
        },
        {
            'entity_type': 'PERSON',
            'entity_value': 'John Smith',
            'context_label': 'educational',
            'confidence': 0.80,
            'is_pii': False,
            'text': 'Name format: FirstName LastName (e.g., John Smith)'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        print(f"Text: {case['text']}")
        print(f"Entity: {case['entity_type']} ('{case['entity_value']}')")
        print(f"Context: {case['context_label']} ({case['confidence']:.0%})")
        print(f"Decision: {'PII' if case['is_pii'] else 'NOT PII'}")
        print()
        
        reasoning = generator.generate(
            entity_type=case['entity_type'],
            entity_value=case['entity_value'],
            context_label=case['context_label'],
            context_confidence=case['confidence'],
            is_pii=case['is_pii'],
            original_text=case['text']
        )
        
        print("AI Reasoning:")
        print(reasoning)