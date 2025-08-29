"""Test suite for PDF report generator."""

import pytest
import tempfile
from pathlib import Path
from src.services.pdf_generator import PDFReportGenerator

class TestPDFReportGenerator:
    """Test PDF report generation."""
    
    @pytest.fixture
    def sample_summary_data(self):
        """Create sample summary data for PDF generation."""
        return {
            "processing": {
                "rows_original": 5000,
                "rows_processed": 4850,
                "rows_removed": 150,
            },
            "cleaning": {
                "duplicates_removed": 150,
                "fields_normalized": ["email", "phone", "name"],
            },
            "categorization": {
                "enabled": True,
                "rows_processed": 4850,
                "distribution": {"lead": 2000, "customer": 1500, "vendor": 1000, "other": 350},
                "sources": {"hf": 3880, "openai": 970},
                "costs": {
                    "model": "gpt-4o-mini",
                    "assumptions": {
                        "input_tokens_per_row": 50,
                        "output_tokens_per_row": 10,
                        "price_per_1k": {"input": 0.00015, "output": 0.0006}
                    },
                    "cost_all_openai": 0.3278,
                    "cost_hybrid": 0.0656,
                    "savings_absolute": 0.2622,
                    "savings_pct": 0.80,
                    "rows_breakdown": {"total": 4850, "hf": 3880, "openai": 970}
                }
            },
            "savings_summary": "Using hybrid categorization saved $0.26 (~80%) compared to sending all rows to OpenAI (model=gpt-4o-mini).",
            "validation": {
                "enabled": True,
                "stats": {
                    "invalid_email": 25,
                    "missing_phone": 15,
                    "invalid_date": 5,
                    "cross_company_phone_duplicate": 8,
                    "numeric_outlier": 12
                }
            }
        }
    
    def test_pdf_generation(self, sample_summary_data):
        """Test basic PDF generation."""
        generator = PDFReportGenerator()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            result_path = generator.generate_report(
                summary_data=sample_summary_data,
                output_path=tmp_file.name,
                original_filename="test_data.csv"
            )
            
            assert result_path == tmp_file.name
            assert Path(tmp_file.name).exists()
            assert Path(tmp_file.name).stat().st_size > 0
    
    def test_infrastructure_recommendations(self):
        """Test infrastructure recommendations generation."""
        generator = PDFReportGenerator()
        
        # Test different scale recommendations
        recommendations_small = generator.create_infrastructure_recommendations(250)
        recommendations_medium = generator.create_infrastructure_recommendations(1000)
        recommendations_large = generator.create_infrastructure_recommendations(5000)
        
        # Small scale shouldn't require Kubernetes
        assert "docker compose" in recommendations_small["scaling"].lower()
        
        # Medium scale should require load balancer
        assert "required" in recommendations_medium["load_balancer"].lower()
        
        # Large scale should have additional services
        assert "cdn" in recommendations_large
        assert "monitoring" in recommendations_large
    
    def test_cost_savings_chart_creation(self, sample_summary_data):
        """Test cost savings chart creation."""
        generator = PDFReportGenerator()
        costs_data = sample_summary_data["categorization"]["costs"]
        
        chart_b64 = generator.create_cost_savings_chart(costs_data)
        
        assert isinstance(chart_b64, str)
        assert len(chart_b64) > 0
        
        # Should be valid base64
        import base64
        try:
            base64.b64decode(chart_b64)
        except Exception:
            pytest.fail("Generated chart is not valid base64")