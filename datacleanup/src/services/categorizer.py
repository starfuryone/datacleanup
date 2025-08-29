"""Enhanced categorizer with Hybrid AI (Hugging Face + OpenAI fallback) and cost tracking."""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import openai

logger = logging.getLogger(__name__)

class HybridCategorizer:
    """Hybrid categorizer using Hugging Face first, OpenAI as fallback."""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        model_name: str = "facebook/bart-large-mnli",
        confidence_threshold: float = 0.7,
    ):
        self.openai_api_key = openai_api_key
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        
        # Initialize Hugging Face classifier
        try:
            self.hf_classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            self.hf_available = True
            logger.info(f"Hugging Face model {model_name} loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Hugging Face model: {e}")
            self.hf_classifier = None
            self.hf_available = False
        
        # Initialize OpenAI client if API key provided
        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_available = True
        else:
            self.openai_available = False
            logger.warning("OpenAI API key not provided - fallback unavailable")
    
    def _classify_with_hf(self, text: str, categories: List[str]) -> Tuple[str, float]:
        """Classify text using Hugging Face zero-shot classification."""
        if not self.hf_available:
            return "other", 0.0
        
        try:
            result = self.hf_classifier(text, categories)
            category = result['labels'][0]
            confidence = result['scores'][0]
            return category, confidence
        except Exception as e:
            logger.warning(f"Hugging Face classification failed: {e}")
            return "other", 0.0
    
    async def _classify_with_openai(self, text: str, categories: List[str]) -> Tuple[str, float]:
        """Classify text using OpenAI as fallback."""
        if not self.openai_available:
            return "other", 0.0
        
        try:
            prompt = f"""Classify the following data entry into one of these categories: {', '.join(categories)}