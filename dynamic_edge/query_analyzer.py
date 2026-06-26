import os
from enum import Enum
from transformers import pipeline

class TemporalClass(Enum):
    STATIC_SCENE = "static"
    LONG_DURATION = "long"
    FAST_ACTION = "fast"

class QueryAnalyzer:
    def __init__(self, model_name="typeform/distilbert-base-uncased-mnli"):
        """
        Initializes the zero-shot classifier used for predicting temporal granularity.
        We use a lightweight distilbert model suitable for edge inference.
        """
        print(f"Loading query analyzer model: {model_name}...")
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        
        # Define candidate labels mapping to our TemporalClass
        self.candidate_labels = [
            "a static scene, a state, counting objects, color, weather, or standing still",
            "a long duration event, a summary, a meeting, or a slow process over time",
            "a fast action, sudden movement, dropping, falling, running, or quick change"
        ]
        
    def analyze(self, query: str) -> TemporalClass:
        """
        Analyzes the query to determine its temporal granularity.
        """
        result = self.classifier(query, self.candidate_labels)
        top_label = result['labels'][0]
        
        if "fast action" in top_label:
            return TemporalClass.FAST_ACTION
        elif "static scene" in top_label:
            return TemporalClass.STATIC_SCENE
        else:
            return TemporalClass.LONG_DURATION

if __name__ == "__main__":
    analyzer = QueryAnalyzer()
    queries = [
        "What color is the car parked outside?",
        "Did the person drop the bag?",
        "Summarize the conversation in this 10 minute video."
    ]
    for q in queries:
        print(f"Query: '{q}'\n-> Class: {analyzer.analyze(q).name}\n")
