"""Test suite for hybrid AI cost savings calculations."""

import pytest
from src.config.costs import calculate_costs, OPENAI_PRICING, TOKEN_ASSUMPTIONS

class TestCostCalculations:
    """Test cost calculation functions."""
    
    def test_calculate_costs_basic(self):
        """Test basic cost calculation."""
        costs = calculate_costs(
            rows_total=1000,
            rows_openai=200,
            ai_pricing_model="gpt-4o-mini",
            input_tokens_per_row=50,
            output_tokens_per_row=10,
        )
        
        # Expected calculations:
        # cost_per_row = (50/1000 * 0.00015) + (10/1000 * 0.0006) = 0.0000135
        # cost_all_openai = 1000 * 0.0000135 = 0.0135
        # cost_hybrid = 200 * 0.0000135 = 0.0027
        # savings = 0.0135 - 0.0027 = 0.0108
        
        assert costs["model"] == "gpt-4o-mini"
        assert costs["cost_all_openai"] == 0.0135
        assert costs["cost_hybrid"] == 0.0027
        assert costs["savings_absolute"] == 0.0108
        assert costs["savings_pct"] == 0.8  # 80%
        assert costs["rows_breakdown"]["total"] == 1000
        assert costs["rows_breakdown"]["hf"] == 800
        assert costs["rows_breakdown"]["openai"] == 200
    
    def test_calculate_costs_all_hf(self):
        """Test cost calculation with all HF (no OpenAI fallback)."""
        costs = calculate_costs(
            rows_total=1000,
            rows_openai=0,
            ai_pricing_model="gpt-4o-mini",
        )
        
        assert costs["cost_hybrid"] == 0.0
        assert costs["savings_pct"] == 1.0  # 100% savings
        assert costs["rows_breakdown"]["hf"] == 1000
        assert costs["rows_breakdown"]["openai"] == 0
    
    def test_calculate_costs_all_openai(self):
        """Test cost calculation with all OpenAI (no HF)."""
        costs = calculate_costs(
            rows_total=1000,
            rows_openai=1000,
            ai_pricing_model="gpt-4o-mini",
        )
        
        assert costs["cost_hybrid"] == costs["cost_all_openai"]
        assert costs["savings_absolute"] == 0.0
        assert costs["savings_pct"] == 0.0  # 0% savings
        assert costs["rows_breakdown"]["hf"] == 0
        assert costs["rows_breakdown"]["openai"] == 1000
    
    def test_calculate_costs_different_models(self):
        """Test cost calculation with different OpenAI models."""
        costs_mini = calculate_costs(
            rows_total=1000,
            rows_openai=500,
            ai_pricing_model="gpt-4o-mini",
        )
        
        costs_4o = calculate_costs(
            rows_total=1000,
            rows_openai=500,
            ai_pricing_model="gpt-4o",
        )
        
        # gpt-4o should be more expensive than gpt-4o-mini
        assert costs_4o["cost_all_openai"] > costs_mini["cost_all_openai"]
        assert costs_4o["cost_hybrid"] > costs_mini["cost_hybrid"]
        assert costs_4o["savings_absolute"] > costs_mini["savings_absolute"]
    
    def test_calculate_costs_invalid_model(self):
        """Test cost calculation with invalid model falls back to default."""
        costs = calculate_costs(
            rows_total=1000,
            rows_openai=500,
            ai_pricing_model="invalid-model",
        )
        
        # Should fallback to gpt-4o-mini
        assert costs["model"] == "gpt-4o-mini"
        assert "cost_all_openai" in costs
        assert costs["cost_all_openai"] > 0
    
    def test_calculate_costs_edge_cases(self):
        """Test cost calculation edge cases."""
        # Zero rows
        costs_zero = calculate_costs(
            rows_total=0,
            rows_openai=0,
        )
        assert costs_zero["cost_all_openai"] == 0.0
        assert costs_zero["cost_hybrid"] == 0.0
        assert costs_zero["savings_pct"] == 0.0
        
        # Single row
        costs_one = calculate_costs(
            rows_total=1,
            rows_openai=1,
        )
        assert costs_one["cost_all_openai"] == costs_one["cost_hybrid"]
        assert costs_one["savings_absolute"] == 0.0