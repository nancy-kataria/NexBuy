"""
Tests for data preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nexbuy.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_init(self):
        """Test initialization."""
        preprocessor = DataPreprocessor(verbose=False)
        assert preprocessor.dataset_name == "vivek468/superstore-dataset-final"
        assert preprocessor.verbose == False
        assert preprocessor.superstore_data is None
    
    @patch('nexbuy.data.preprocessing.kagglehub.dataset_download')
    @patch('nexbuy.data.preprocessing.pd.read_csv')
    def test_download_and_load_data_success(self, mock_read_csv, mock_download):
        """Test successful data loading."""
        # Mock the download and read operations
        mock_download.return_value = "/fake/path"
        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_read_csv.return_value = mock_df
        
        preprocessor = DataPreprocessor(verbose=False)
        result = preprocessor.download_and_load_data()
        
        assert result.equals(mock_df)
        assert preprocessor.superstore_data.equals(mock_df)
    
    def test_clean_data_without_loading(self):
        """Test that clean_data raises error if data not loaded."""
        preprocessor = DataPreprocessor(verbose=False)
        
        with pytest.raises(ValueError, match="Data must be loaded first"):
            preprocessor.clean_data()
    
    def test_split_data_without_processing(self):
        """Test that split_data raises error if data not processed."""
        preprocessor = DataPreprocessor(verbose=False)
        
        with pytest.raises(ValueError, match="Data must be processed first"):
            preprocessor.split_data()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Order ID': ['A1', 'A2', 'A3', 'A4'],
        'Order Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Ship Date': ['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'Customer ID': ['C1', 'C2', 'C1', 'C2'],
        'Product ID': ['P1', 'P2', 'P3', 'P4'],
        'Product Name': ['Product 1', 'Product 2', 'Product 3', 'Product 4'],
        'Sales': [100, 200, 150, 300],
        'Quantity': [1, 2, 1, 3],
        'Category': ['Cat1', 'Cat2', 'Cat1', 'Cat2'],
        'Sub-Category': ['Sub1', 'Sub2', 'Sub1', 'Sub2']
    })


class TestDataPreprocessorWithData:
    """Test cases with actual data."""
    
    def test_clean_data_with_sample(self, sample_data):
        """Test clean_data with sample data."""
        preprocessor = DataPreprocessor(verbose=False)
        preprocessor.superstore_data = sample_data.copy()
        
        result = preprocessor.clean_data()
        
        # Check that dates are converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(result['Order Date'])
        assert pd.api.types.is_datetime64_any_dtype(result['Ship Date'])
        
        # Check that data is not empty
        assert not result.empty
    
    def test_split_data_with_sample(self, sample_data):
        """Test split_data with sample data."""
        preprocessor = DataPreprocessor(verbose=False)
        preprocessor.superstore_data = sample_data.copy()
        preprocessor.superstore_data['Order Date'] = pd.to_datetime(preprocessor.superstore_data['Order Date'])
        
        train_df, test_df = preprocessor.split_data(train_ratio=0.75)
        
        # Check split proportions (roughly)
        total_rows = len(sample_data)
        assert len(train_df) == int(total_rows * 0.75)
        assert len(test_df) == total_rows - len(train_df)
        
        # Check that train and test don't overlap
        assert len(pd.concat([train_df, test_df])) == total_rows


if __name__ == "__main__":
    pytest.main([__file__])
