import pytest
import numpy as np
import pandas as pd
import warnings
from metrics import validate_inputs

class TestValidateInputsDirty:
    
    def test_different_length_arrays(self):
        """Test that validate_inputs raises ValueError when arrays have different lengths."""
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 3]
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            validate_inputs(y_true, y_pred)
    
    def test_empty_arrays(self):
        """Test that validate_inputs raises ValueError when arrays are empty."""
        y_true = []
        y_pred = []
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            validate_inputs(y_true, y_pred)
    
    def test_string_inputs(self):
        """Test that validate_inputs raises TypeError when string inputs are provided."""
        y_true = "not an array"
        y_pred = [1, 2, 3]
        with pytest.raises(TypeError, match="String inputs are not supported"):
            validate_inputs(y_true, y_pred)
    
    def test_non_numeric_values(self):
        """Test that validate_inputs raises ValueError when arrays contain non-numeric values."""
        y_true = [1, 2, "three", 4]
        y_pred = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Failed to convert y_true to float array"):
            validate_inputs(y_true, y_pred)
    
    def test_nan_values(self):
        """Test that validate_inputs raises ValueError when arrays contain NaN values."""
        y_true = [1, 2, np.nan, 4]
        y_pred = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Input arrays contain NaN values"):
            validate_inputs(y_true, y_pred)
    
    def test_infinity_values(self):
        """Test that validate_inputs raises ValueError when arrays contain infinity values."""
        y_true = [1, 2, np.inf, 4]
        y_pred = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="Input arrays contain infinity values"):
            validate_inputs(y_true, y_pred)
    
    def test_warning_all_zeros_true(self):
        """Test that validate_inputs warns when all y_true values are zero."""
        y_true = [0, 0, 0, 0]
        y_pred = [1, 2, 3, 4]
        with pytest.warns(UserWarning, match="All values in y_true are zero"):
            validate_inputs(y_true, y_pred)
    
    def test_warning_all_zeros_pred(self):
        """Test that validate_inputs warns when all y_pred values are zero."""
        y_true = [1, 2, 3, 4]
        y_pred = [0, 0, 0, 0]
        with pytest.warns(UserWarning, match="All values in y_pred are zero"):
            validate_inputs(y_true, y_pred)
    
    def test_warning_large_differences(self):
        """Test that validate_inputs warns when there are very large differences between arrays."""
        y_true = [1, 2, 3, 4]
        y_pred = [1, 2, 3, 1e7]
        with pytest.warns(UserWarning, match="Very large differences detected"):
            validate_inputs(y_true, y_pred)
    
    def test_warning_multidimensional_array(self):
        """Test that validate_inputs warns when a multi-dimensional array is provided."""
        y_true = np.array([[1, 2], [3, 4], [5, 6]])  # Flattens to 6 elements
        y_pred = [1, 2, 3, 4, 5, 6]  # Now has 6 elements
        with pytest.warns(UserWarning, match="Multi-dimensional array provided"):
            validate_inputs(y_true, y_pred)


class TestValidateInputsFunctionality:
    
    def test_list_inputs(self):
        """Test validate_inputs with basic list inputs."""
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
        """Test validate_inputs with numpy array inputs."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([2, 3, 4, 5])
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2, 3, 4, 5]))
    
    def test_pandas_series_inputs(self):
        """Test validate_inputs with pandas Series inputs."""
        y_true = pd.Series([1, 2, 3, 4])
        y_pred = pd.Series([2, 3, 4, 5])
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2, 3, 4, 5]))
    
    def test_pandas_dataframe_inputs(self):
        """Test validate_inputs with pandas DataFrame inputs (single column)."""
        y_true = pd.DataFrame({'value': [1, 2, 3, 4]})
        y_pred = pd.DataFrame({'value': [2, 3, 4, 5]})
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2, 3, 4, 5]))
    
    def test_scalar_inputs(self):
        """Test validate_inputs with scalar inputs."""
        y_true = 5
        y_pred = 7
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([5]))
        np.testing.assert_array_equal(y_pred_arr, np.array([7]))
    
    def test_mixed_inputs(self):
        """Test validate_inputs with mixed input types."""
        y_true = [1, 2, 3, 4]
        y_pred = pd.Series([2, 3, 4, 5])
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2, 3, 4, 5]))
    
    def test_multidimensional_array_flattening(self):
        """Test validate_inputs correctly flattens multi-dimensional arrays."""
        y_true = np.array([[1, 2], [3, 4]])
        y_pred = np.array([1, 2, 3, 4])
        
        with pytest.warns(UserWarning, match="Multi-dimensional array provided"):
            y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred_arr, np.array([1, 2, 3, 4]))
    
    def test_multidimensional_dataframe_flattening(self):
        """Test validate_inputs correctly flattens multi-dimensional DataFrames."""
        y_true = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        y_pred = np.array([1, 2, 3, 4])
        
        with pytest.warns(UserWarning, match="Multi-dimensional DataFrame provided"):
            y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        # The exact order depends on how pandas flattens the DataFrame
        assert set(y_true_arr) == {1, 2, 3, 4}
        np.testing.assert_array_equal(y_pred_arr, np.array([1, 2, 3, 4]))
    
    def test_float_conversion(self):
        """Test validate_inputs correctly converts values to float."""
        y_true = [1, 2, 3, 4]
        y_pred = ["2.0", "3.0", "4.0", "5.0"]  # String numbers should convert to float
        y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
        
        np.testing.assert_array_equal(y_true_arr, np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(y_pred_arr, np.array([2.0, 3.0, 4.0, 5.0]))
        
        # Check that the dtype is float
        assert y_true_arr.dtype == np.float64
        assert y_pred_arr.dtype == np.float64
