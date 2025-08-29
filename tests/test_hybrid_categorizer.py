"""Test suite for hybrid AI categorizer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import pandas as pd
from src.services.categorizer import HybridCategorizer

class TestHybridCategorizer:
    """Test hybrid AI categorizer functionality."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            'name': ['John Doe', 'Jane Smith', 'ACME Corp'],
            'email': ['john@email.com', 'jane@email.com', 'info@acme.com'],
            'company': ['ABC Inc', 'XYZ Corp', 'ACME Corp'],
            'description': ['Software engineer', 'Marketing manager', 'Corporate supplier']
        })
    
    @pytest.fixture
    def mock_categorizer_hf_only(self):
        """Create categorizer with only HF available."""
        with patch('src.services.categorizer.pipeline') as mock_pipeline:
            mock_classifier = MagicMock()
            mock_classifier.return_value = {
                'labels': ['lead', 'customer'],
                'scores': [0.8, 0.2]
            }
            mock_pipeline.return_value = mock_classifier
            
            categorizer = HybridCategorizer(openai_api_key=None)
            yield categorizer
    
    @pytest.fixture  
    def mock_categorizer_hybrid(self):
        """Create categorizer with both HF and OpenAI."""
        with patch('src.services.categorizer.pipeline') as mock_pipeline, \
             patch('src.services.categorizer.openai') as mock_openai:
            
            # Mock HF classifier
            mock_classifier = MagicMock()
            mock_classifier.return_value = {
                'labels': ['lead', 'customer'],
                'scores': [0.6, 0.4]  # Low confidence to trigger fallback
            }
            mock_pipeline.return_value = mock_classifier
            
            # Mock OpenAI response
            mock_response = AsyncMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "customer"
            mock_openai.ChatCompletion.acreate.return_value = mock_response
            
            categorizer = HybridCategorizer(openai_api_key="test-key")
            yield categorizer
    
    @pytest.mark.asyncio
    async def test_categorize_hf_only(self, mock_categorizer_hf_only, sample_dataframe):
        """Test categorization using only Hugging Face."""
        categories = ["lead", "customer", "vendor"]
        
        df_result, summary = await mock_categorizer_hf_only.categorize_dataframe(
            sample_dataframe, categories, ai_model="hf"
        )
        
        assert len(df_result) == len(sample_dataframe)
        assert 'category' in df_result.columns
        assert 'category_confidence' in df_result.columns
        assert 'source_model' in df_result.columns
        assert all(df_result['source_model'] == 'hf')
        
        assert summary["enabled"] is True
        assert summary["rows_processed"] == len(sample_dataframe)
        assert summary["sources"]["hf"] == len(sample_dataframe)
        assert summary["sources"]["openai"] == 0
    
    @pytest.mark.asyncio
    async def test_categorize_hybrid_fallback(self, mock_categorizer_hybrid, sample_dataframe):
        """Test hybrid categorization with OpenAI fallback."""
        categories = ["lead", "customer", "vendor"]
        
        df_result, summary = await mock_categorizer_hybrid.categorize_dataframe(
            sample_dataframe, categories, ai_model="hybrid"
        )
        
        assert len(df_result) == len(sample_dataframe)
        assert 'source_model' in df_result.columns
        
        # Should have some OpenAI fallbacks due to low confidence
        assert summary["sources"]["openai"] > 0
        
        # Check confidence bins
        assert "confidence_bins" in summary
        assert summary["confidence_bins"]["high"] >= 0
    
    @pytest.mark.asyncio
    async def test_categorize_empty_dataframe(self, mock_categorizer_hf_only):
        """Test categorization with empty dataframe."""
        empty_df = pd.DataFrame()
        categories = ["lead", "customer"]
        
        df_result, summary = await mock_categorizer_hf_only.categorize_dataframe(
            empty_df, categories
        )
        
        assert len(df_result) == 0
        assert summary["enabled"] is False
        assert summary["rows_processed"] == 0