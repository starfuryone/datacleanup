"""Integration tests for enhanced API endpoints."""

import pytest
from fastapi.testclient import TestClient
import json
import io
from src.app import app

client = TestClient(app)

class TestAPIIntegration:
    """Test API integration with cost savings features."""
    
    def test_cost_estimate_endpoint(self):
        """Test cost estimation endpoint."""
        response = client.get(
            "/v1/cost-estimate",
            params={
                "rows": 1000,
                "ai_model": "hybrid",
                "ai_pricing_model": "gpt-4o-mini",
                "ai_input_tokens": 50,
                "ai_output_tokens": 10,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "estimate" in data
        assert "savings_summary" in data
        assert "parameters" in data
        assert data["parameters"]["rows"] == 1000
        
        estimate = data["estimate"]
        assert "cost_all_openai" in estimate
        assert "cost_hybrid" in estimate
        assert "savings_absolute" in estimate
        assert estimate["savings_pct"] >= 0
    
    def test_infrastructure_requirements_endpoint(self):
        """Test infrastructure requirements endpoint."""
        response = client.get(
            "/v1/infrastructure-requirements",
            params={
                "concurrent_users": 1000,
                "include_costs": True,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "concurrent_users" in data
        assert "server_requirements" in data
        assert "scaling" in data
        assert data["concurrent_users"] == 1000
        
        server_req = data["server_requirements"]
        assert "cpu_cores" in server_req
        assert "ram" in server_req
        assert "storage" in server_req
        
        # Should include cost estimates
        assert "cost_estimates" in data
        assert "monthly_usd" in data["cost_estimates"]
    
    @pytest.mark.asyncio
    async def test_clean_with_cost_analysis(self):
        """Test clean endpoint with cost analysis."""
        # Create sample CSV data
        csv_data = """name,email,company