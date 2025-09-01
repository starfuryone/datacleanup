from __future__ import annotations
import os
from typing import Dict

# OpenAI pricing per 1K tokens (as of Jan 2025)
OPENAI_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}

# Token usage assumptions per row (overridable at runtime)
TOKEN_ASSUMPTIONS = {
    "input_tokens_per_row": 50,
    "output_tokens_per_row": 10,
}

def get_cost_config() -> Dict:
    """Get cost configuration with environment overrides."""
    return {
        "ai_pricing_model": os.getenv("AI_PRICING_MODEL", "gpt-4o-mini"),
        "ai_input_tokens_per_row": int(os.getenv("AI_INPUT_TOKENS_PER_ROW", "50")),
        "ai_output_tokens_per_row": int(os.getenv("AI_OUTPUT_TOKENS_PER_ROW", "10")),
    }

def calculate_costs(
    rows_total: int,
    rows_openai: int,
    ai_pricing_model: str = "gpt-4o-mini",
    input_tokens_per_row: int = 50,
    output_tokens_per_row: int = 10,
) -> Dict:
    """Calculate cost savings from Hybrid AI vs All-OpenAI approach."""
    if ai_pricing_model not in OPENAI_PRICING:
        ai_pricing_model = "gpt-4o-mini"  # fallback

    pricing = OPENAI_PRICING[ai_pricing_model]
    rows_hf = rows_total - rows_openai

    # Cost per row calculation
    cost_per_row = (
        input_tokens_per_row / 1000.0 * pricing["input"] +
        output_tokens_per_row / 1000.0 * pricing["output"]
    )

    # Cost scenarios
    cost_all_openai = rows_total * cost_per_row
    cost_hybrid = rows_openai * cost_per_row  # HF assumed $0 variable cost

    # Savings calculation
    savings_absolute = cost_all_openai - cost_hybrid
    savings_pct = 0.0 if cost_all_openai == 0 else (savings_absolute / cost_all_openai)

    return {
        "model": ai_pricing_model,
        "assumptions": {
            "input_tokens_per_row": input_tokens_per_row,
            "output_tokens_per_row": output_tokens_per_row,
            "price_per_1k": {"input": pricing["input"], "output": pricing["output"]},
        },
        "cost_all_openai": round(cost_all_openai, 4),
        "cost_hybrid": round(cost_hybrid, 4),
        "savings_absolute": round(savings_absolute, 4),
        "savings_pct": round(savings_pct, 2),
        "rows_breakdown": {
            "total": rows_total,
            "hf": rows_hf,
            "openai": rows_openai,
        },
    }
