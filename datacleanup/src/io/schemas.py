from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AICfg(BaseModel):
    enabled: bool = True
    model: str = "hybrid"
    categories: List[str] = ["lead","customer","vendor","expense","other"]

class CleanConfig(BaseModel):
    ai: AICfg = AICfg()
    validate: bool = False
    dedupe_keys: List[str] = ["email", "company", "phone"]

class CostSummary(BaseModel):
    model: str
    cost_all_openai: float
    cost_hybrid: float
    savings_absolute: float
    savings_pct: float
    rows_breakdown: Dict[str, int]

class ValidationStats(BaseModel):
    enabled: bool = False
    stats: Dict[str, int] = Field(default_factory=dict)
    samples: Dict[str, Any] = Field(default_factory=dict)

class CategorizationSummary(BaseModel):
    enabled: bool = False
    rows_processed: int = 0
    distribution: Dict[str, int] = Field(default_factory=dict)
    confidence_bins: Dict[str, int] = Field(default_factory=dict)
    sources: Dict[str, int] = Field(default_factory=dict)
    costs: Optional[CostSummary] = None

class Summary(BaseModel):
    rows_processed: int = 0
    duplicates_removed: int = 0
    data_quality_score: float = 0.0
    categorization: CategorizationSummary = CategorizationSummary()
    validation: ValidationStats = ValidationStats()
