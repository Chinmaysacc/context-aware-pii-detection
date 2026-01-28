"""
Layer 2: Context Classification using Machine Learning

This module classifies text context to determine if entities are
real PII or just examples/demonstrations.

Context Types:
- personal_data: Real user information
- example_demo: Illustrative samples
- educational: Format descriptions
- public_info: Publicly available data
"""

import re
from typing import Dict, Tuple, List
import pickle
import os
from dataclasses import dataclass


@dataclass
class ContextResult:
    """Context classification result"""
    label: str
    confidence: float
    features: Dict[str, int]


class ContextClassifier:
    """
    Rule-based context classifier (baseline version)
    
    Uses keyword patterns to classify text context.
    """
    
    def __init__(self):
        # Context feature patterns
        self.patterns = {
            'personal_data': [
                r'\bmy\b', r'\bi am\b', r'\bme\b', r'\bmine\b',
                r'\bcontact me\b', r'\breach me\b', r'\bcall me\b',
                r'\bemail me\b', r'\btext me\b',
                r'\bmy (name|email|phone|address|number|card|ssn) is\b',
                r'\byou can (reach|contact|call|email) me\b'
            ],
            'example_demo': [
                r'\bexample\b', r'\bfor instance\b', r'\be\.?g\.?\b',
                r'\bsuch as\b', r'\blike\b', r'\bsample\b',
                r'\bfor example\b', r'\bas an example\b',
                r'\bexample of\b', r'\binstances? include\b'
            ],
            'educational': [
                r'\bformat\b', r'\bstructure\b', r'\bpattern\b',
                r'\bshould be\b', r'\bmust be\b', r'\btemplate\b',
                r'\blooks like\b', r'\bin the form\b',
                r'\bfollows? the pattern\b', r'\bshape of\b'
            ],
            'public_info': [
                r'\bcompany\b', r'\borganization\b', r'\bwebsite\b',
                r'\bpublic\b', r'\bofficial\b', r'\bcorporate\b',
                r'\bpress release\b', r'\bpublicly available\b'
            ]
        }
    
    def extract_features(self, text: str) -> Dict[str, int]:
        """
        Extract feature counts for each context type
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of context type to feature count
        """
        features = {}
        text_lower = text.lower()
        
        for context, patterns in self.patterns.items():
            count = sum(
                1 for pattern in patterns 
                if re.search(pattern, text_lower, re.IGNORECASE)
            )
            features[context] = count
        
        return features
    
    def classify(self, text: str) -> ContextResult:
        """
        Classify text context using feature-based voting
        
        Args:
            text: Input text to classify
            
        Returns:
            ContextResult with label, confidence, and features
        """
        features = self.extract_features(text)
        
        # Find dominant context
        max_count = max(features.values())
        
        if max_count == 0:
            # No clear context detected
            return ContextResult(
                label='unknown',
                confidence=0.5,
                features=features
            )
        
        # Get context with highest score
        predicted_context = max(features, key=features.get)
        
        # Calculate confidence (normalize by max possible score)
        confidence = min(max_count / 5.0, 0.95)
        
        return ContextResult(
            label=predicted_context,
            confidence=confidence,
            features=features
        )


class MLContextClassifier:
    """
    Machine Learning-based context classifier
    
    Uses TF-IDF + Logistic Regression for classification.
    Can be trained on labeled data for better accuracy.
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def train(self, X_train: List[str], y_train: List[str]):
        """
        Train the ML model
        
        Args:
            X_train: List of training texts
            y_train: List of context labels
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.pipeline import Pipeline  # type: ignore
            
            # Create pipeline
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 3),
                    stop_words='english'
                )),
                ('classifier', LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            
            # Train
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            print(f"✓ Model trained on {len(X_train)} examples")
            
        except ImportError as e:
            print(f"⚠ scikit-learn not installed ({e}). Using rule-based fallback.")
            print("  To install: pip install scikit-learn")
            self.fallback_classifier = ContextClassifier()
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict context label and confidence
        
        Args:
            text: Input text
            
        Returns:
            (label, confidence) tuple
        """
        if self.is_trained and self.model:
            label = self.model.predict([text])[0]
            probabilities = self.model.predict_proba([text])[0]
            confidence = max(probabilities)
            return label, confidence
        
        elif hasattr(self, 'fallback_classifier'):
            result = self.fallback_classifier.classify(text)
            return result.label, result.confidence
        
        else:
            return 'unknown', 0.5
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        if self.model:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"✓ Model loaded from {path}")


def create_training_data() -> Tuple[List[str], List[str]]:
    """
    Create sample training dataset
    
    Returns:
        (X_train, y_train) tuple
    """
    X_train = [
        # Personal data examples (20)
        "My name is John Smith",
        "Contact me at john@email.com",
        "You can reach me at 555-1234",
        "I am Sarah Jones, call me at 555-9876",
        "Email me at sarah@company.com",
        "My phone number is +1-555-123-4567",
        "Text me at 555-0000",
        "You can contact me via email at info@mysite.com",
        "My SSN is 123-45-6789",
        "Reach me on my mobile: 555-7777",
        "I can be reached at john.doe@gmail.com",
        "Call me anytime at +1-555-999-8888",
        "My credit card number is 4532-1111-2222-3333",
        "Contact details: 555-4321",
        "You can email me directly at contact@me.com",
        "My address is 123 Main Street",
        "Text or call me at 555-5555",
        "I'm available at myemail@domain.com",
        "My driver's license number is D1234567",
        "Reach out to me at work: 555-0001",
        
        # Example/demo data (20)
        "Example of an email: user@example.com",
        "For instance, a phone number might look like 555-0000",
        "Such as: john@sample.com",
        "Sample phone: 555-1234",
        "Examples include: name@domain.com",
        "Like this: +1-555-000-0000",
        "For example: 123-45-6789",
        "E.g., user123@test.com",
        "Example format: 555-1111",
        "Sample email address: test@test.com",
        "For instance: John Doe",
        "Such as a credit card: 4532-0000-0000-0000",
        "Example SSN: 000-00-0000",
        "Like 192.168.1.1",
        "Sample name: Jane Smith",
        "For example, addresses like user@site.com",
        "Examples of valid emails: admin@server.com",
        "Such as phone numbers: 555-9999",
        "Instance: john.smith@email.com",
        "Sample data: 555-8888",
        
        # Educational data (15)
        "Email format should be: username@domain.extension",
        "Phone numbers must follow the pattern: (XXX) XXX-XXXX",
        "The structure is: FirstName LastName",
        "Format template: name@company.com",
        "Pattern for SSN: XXX-XX-XXXX",
        "Phone format: +1-XXX-XXX-XXXX",
        "Email structure looks like: local@domain.tld",
        "The pattern should be: XXXX-XXXX-XXXX-XXXX",
        "Format must be in the form: +country-area-number",
        "Structure follows: prefix@domain.suffix",
        "Template for addresses: number street city",
        "The format is: three digits, dash, two digits, dash, four digits",
        "Phone pattern: area code followed by seven digits",
        "Email must be in username@domain format",
        "The shape of a credit card: 16 digits in 4 groups",
        
        # Public info (10)
        "Company website: info@company.com",
        "Corporate contact: contact@organization.org",
        "Official email: press@business.com",
        "Public inquiry line: 555-CORP",
        "Organization's main number: 1-800-555-0000",
        "Website contact form: www.company.com/contact",
        "Press release contact: media@corp.com",
        "Publicly available: support@service.com",
        "Corporate headquarters: 555-HEAD",
        "Official support: help@platform.com",
        
        # Ambiguous/unknown (10)
        "The number is 555-1234",
        "Email: john@email.com",
        "Contact: 555-9999",
        "Address: test@test.com",
        "Number: +1-555-000-0000",
        "555-7777",
        "john.doe@sample.com",
        "Phone: 555-4444",
        "123-45-6789",
        "user@domain.com"
    ]
    
    y_train = (
        ['personal_data'] * 20 +
        ['example_demo'] * 20 +
        ['educational'] * 15 +
        ['public_info'] * 10 +
        ['unknown'] * 10
    )
    
    return X_train, y_train


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("CONTEXT CLASSIFICATION DEMO")
    print("=" * 80)
    
    # Test rule-based classifier
    print("\n1. Rule-Based Classifier:")
    classifier = ContextClassifier()
    
    test_texts = [
        "My phone number is 555-1234",
        "Example of phone: 555-9999",
        "Email format should be: user@domain.com"
    ]
    
    for text in test_texts:
        result = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"Context: {result.label} (confidence: {result.confidence:.2%})")
        print(f"Features: {result.features}")
    
    # Test ML classifier
    print("\n" + "=" * 80)
    print("2. ML Classifier (Training):")
    
    ml_classifier = MLContextClassifier()
    X_train, y_train = create_training_data()
    ml_classifier.train(X_train, y_train)
    
    print("\nTesting ML Classifier:")
    for text in test_texts:
        label, confidence = ml_classifier.predict(text)
        print(f"\nText: {text}")
        print(f"Context: {label} (confidence: {confidence:.2%})")