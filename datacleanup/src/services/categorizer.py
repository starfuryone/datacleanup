"""Enhanced categorizer with Hybrid AI (Hugging Face + OpenAI fallback) and cost tracking."""
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

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

        self.hf_available = False
        if HF_AVAILABLE:
            try:
                self.hf_classifier = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                )
                self.hf_available = True
                logger.info("Hugging Face model loaded")
            except Exception as e:
                logger.warning(f"Failed to load Hugging Face model: {e}")
                self.hf_classifier = None
        else:
            self.hf_classifier = None

        self.openai_available = False
        if OPENAI_AVAILABLE and openai_api_key:
            try:
                openai.api_key = openai_api_key
                self.openai_available = True
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")

    def _classify_with_hf(self, text: str, categories: List[str]) -> Tuple[str, float]:
        if not self.hf_available or not self.hf_classifier:
            return "other", 0.0
        try:
            result = self.hf_classifier(text, categories)
            category = result['labels'][0]
            confidence = float(result['scores'][0])
            return category, confidence
        except Exception as e:
            logger.warning(f"Hugging Face classification failed: {e}")
            return "other", 0.0

    def _classify_with_openai(self, text: str, categories: List[str]) -> Tuple[str, float]:
        if not self.openai_available:
            return "other", 0.0
        try:
            prompt = f"Classify into one of: {', '.join(categories)}.\nData: {text}\nRespond with only the category."
            # Use Chat Completions if available, else fall back gracefully
            rsp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )
            category = (rsp.choices[0].message.content or "").strip().lower()
            if category not in [c.lower() for c in categories]:
                category = "other"
            return category, 0.9
        except Exception as e:
            logger.error(f"OpenAI classification failed: {e}")
            return "other", 0.0

    def categorize_dataframe(
        self,
        df: pd.DataFrame,
        categories: List[str],
        text_columns: Optional[List[str]] = None,
        ai_model: str = "hybrid",  # "hybrid", "hf", "openai"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df.empty:
            return df, {"enabled": False, "rows_processed": 0}

        if text_columns is None:
            text_columns = [col for col in df.columns 
                            if df[col].dtype == 'object' and col.lower() not in ['id', 'index']]

        if not text_columns:
            df['category'] = 'other'
            df['category_confidence'] = 0.0
            df['source_model'] = 'none'
            return df, {"enabled": False, "rows_processed": 0}

        rows_total = len(df)
        rows_hf = 0
        rows_openai = 0
        distribution = {cat: 0 for cat in categories}
        confidence_bins = {"high": 0, "medium": 0, "low": 0}

        cats, confs, sources = [], [], []

        for _, row in df.iterrows():
            text = " ".join([str(row[c]) for c in text_columns if pd.notna(row[c])])
            category, confidence, source_model = "other", 0.0, "none"

            if ai_model in ["hybrid", "hf"] and self.hf_available:
                category, confidence = self._classify_with_hf(text, categories)
                source_model = "hf"
                rows_hf += 1
                if ai_model == "hybrid" and confidence < self.confidence_threshold and self.openai_available:
                    category, confidence = self._classify_with_openai(text, categories)
                    source_model = "openai"
                    rows_hf -= 1
                    rows_openai += 1
            elif ai_model == "openai" and self.openai_available:
                category, confidence = self._classify_with_openai(text, categories)
                source_model = "openai"
                rows_openai += 1

            cats.append(category)
            confs.append(confidence)
            sources.append(source_model)

            key = category if category in distribution else "other"
            distribution[key] = distribution.get(key, 0) + 1

            if confidence >= 0.8:
                confidence_bins["high"] += 1
            elif confidence >= 0.5:
                confidence_bins["medium"] += 1
            else:
                confidence_bins["low"] += 1

        df['category'] = cats
        df['category_confidence'] = confs
        df['source_model'] = sources

        summary = {
            "enabled": True,
            "rows_processed": rows_total,
            "distribution": distribution,
            "confidence_bins": confidence_bins,
            "sources": {"hf": rows_hf, "openai": rows_openai},
            "model_config": {
                "ai_model": ai_model,
                "hf_model": self.model_name,
                "confidence_threshold": self.confidence_threshold,
            }
        }
        return df, summary
