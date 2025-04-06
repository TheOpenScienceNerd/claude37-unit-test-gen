import pytest
import numpy as np
import pandas as pd
import warnings
from metrics import validate_inputs

class TestValidateInputsFunctionality:
    """Test suite for validate_inputs function core functionality."""
    
    def test_list_inputs(self):
        """Test with basic list inputs."""
        y_true = [1, 2, 3, 4]
        y_pred = [2, 3, 4, 5]
        
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        # Check output types
        assert isinstance(y_true_arr, np.ndarray)
        assert isinstance(y_pred_arr, np.ndarray)
        
        # Check values
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2, 3, 4, 5]))
    
    def test_numpy_array_inputs(self):
        """Test with numpy array inputs."""
        y_true = np.array([5, 6, 7, 8])
        y_pred = np.array([4, 5, 6, 7])
        
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([5, 6, 7, 8]))
        np.testing.assert_array_equal(y_pred_arr, np.array([4, 5, 6, 7]))
    
    def test_pandas_series_inputs(self):
        """Test with pandas Series inputs."""
        y_true = pd.Series([10, 20, 30, 40])
        y_pred = pd.Series([12, 22, 32, 42])
        
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([10, 20, 30, 40]))
        np.testing.assert_array_equal(y_pred_arr, np.array([12, 22, 32, 42]))
    
    def test_scalar_inputs(self):
        """Test with scalar inputs."""
        y_true = 5
        y_pred = 7
        
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([5]))
        np.testing.assert_array_equal(y_pred_arr, np.array([7]))
    
    def test_mixed_input_types(self):
        """Test with mixed input types."""
        y_true = [1, 2, 3, 4]
        y_pred = np.array([2, 3, 4, 5])
        
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2, 3, 4, 5]))
    
    def test_pandas_dataframe_single_column(self):
        """Test with pandas DataFrame with a single column."""
        y_true = pd.DataFrame({'A': [1, 2, 3, 4]})
        y_pred = [2, 3, 4, 5]
        
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2, 3, 4, 5]))
    
    def test_flattening_multidimensional_array(self):
        """Test flattening of multi-dimensional arrays."""
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([[2, 3], [4, 5]])
        
        with pytest.warns(UserWarning, match="Multi-dimensional array provided"):
            y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2, 3, 4, 5]))
    
    def test_float_inputs(self):
        """Test with float inputs."""
        y_true = [1.5, 2.5, 3.5, 4.5]
        y_pred = [1.7, 2.7, 3.7, 4.7]
        
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_almost_equal(y_true_arr, np.array([1.5, 2.5, 3.5, 4.5]))
        np.testing.assert_array_almost_equal(y_pred_arr, np.array([1.7, 2.7, 3.7, 4.7]))
    
    def test_warning_all_zeros_true(self):
        """Test warning when all true values are zero."""
        y_true = [0, 0, 0, 0]
        y_pred = [1, 2, 3, 4]
        
        with pytest.warns(UserWarning, match="All values in y_true are zero"):
            validate_inputs(y_true, y_pred)
    
    def test_warning_all_zeros_pred(self):
        """Test warning when all predicted values are zero."""
        y_true = [1, 2, 3, 4]
        y_pred = [0, 0, 0, 0]
        
        with pytest.warns(UserWarning, match="All values in y_pred are zero"):
            validate_inputs(y_true, y_pred)
    
    def test_warning_large_differences(self):
        """Test warning when there are very large differences between values."""
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 3, 1e7]
        
        with pytest.warns(UserWarning, match="Very large differences detected"):
            validate_inputs(y_true, y_pred)
