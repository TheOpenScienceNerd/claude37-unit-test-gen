import pytest
import numpy as np
import pandas as pd
import warnings
from metrics import validate_inputs, mean_absolute_error

class TestValidateInputs:
    def test_valid_inputs(self):
        """Test with valid inputs in different formats."""
        # List inputs
        y_true, y_pred = validate_inputs([1, 2, 3], [1.5, 2.5, 3.5])
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        np.testing.assert_array_equal(y_true, np.array([1, 2, 3]))
        np.testing.assert_array_equal(y_pred, np.array([1.5, 2.5, 3.5]))
        
        # Numpy array inputs
        y_true, y_pred = validate_inputs(np.array([1, 2, 3]), np.array([1.5, 2.5, 3.5]))
        np.testing.assert_array_equal(y_true, np.array([1, 2, 3]))
        np.testing.assert_array_equal(y_pred, np.array([1.5, 2.5, 3.5]))
        
        # Tuple inputs
        y_true, y_pred = validate_inputs((1, 2, 3), (1.5, 2.5, 3.5))
        np.testing.assert_array_equal(y_true, np.array([1, 2, 3]))
        np.testing.assert_array_equal(y_pred, np.array([1.5, 2.5, 3.5]))
    
    def test_flattening(self):
        """Test that multi-dimensional arrays are flattened correctly."""
        y_true, y_pred = validate_inputs([[1, 2], [3, 4]], [[1.5, 2.5], [3.5, 4.5]])
        np.testing.assert_array_equal(y_true, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(y_pred, np.array([1.5, 2.5, 3.5, 4.5]))
    
    def test_type_error(self):
        """Test that TypeError is raised for non-iterable inputs."""
        with pytest.raises(TypeError, match="Inputs must be iterable"):
            validate_inputs(123, [1, 2, 3])
        
        with pytest.raises(TypeError, match="Inputs must be iterable"):
            validate_inputs([1, 2, 3], 123)
    
    def test_value_error_conversion(self):
        """Test that ValueError is raised when inputs can't be converted to arrays."""
        with pytest.raises(ValueError, match="Inputs cannot be converted to arrays"):
            validate_inputs([1, 2, "a"], [1, 2, 3])
    
    def test_value_error_length(self):
        """Test that ValueError is raised when inputs have different lengths."""
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            validate_inputs([1, 2, 3], [1, 2])
    
    def test_value_error_empty(self):
        """Test that ValueError is raised when inputs are empty."""
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            validate_inputs([], [])
    
    def test_value_error_non_numeric(self):
        """Test that ValueError is raised when inputs contain non-numeric values."""
        with pytest.raises(ValueError, match="Input arrays must contain numeric values"):
            validate_inputs(["1", "2", "3"], ["1", "2", "3"])

class TestMeanAbsoluteError:
    def test_identical_arrays(self):
        """Test MAE with identical arrays (should be zero)."""
        mae = mean_absolute_error([1, 2, 3], [1, 2, 3])
        assert mae == 0.0
    
    def test_basic_calculation(self):
        """Test basic MAE calculation."""
        # MAE = (|1-2| + |2-4| + |3-6|) / 3 = (1 + 2 + 3) / 3 = 2.0
        mae = mean_absolute_error([1, 2, 3], [2, 4, 6])
        assert mae == 2.0
    
    def test_negative_values(self):
        """Test MAE with negative values."""
        # MAE = (|-1-(-2)| + |-2-(-4)| + |-3-(-6)|) / 3 = (1 + 2 + 3) / 3 = 2.0
        mae = mean_absolute_error([-1, -2, -3], [-2, -4, -6])
        assert mae == 2.0
    
    def test_mixed_values(self):
        """Test MAE with mixed positive and negative values."""
        # MAE = (|1-(-1)| + |-2-2| + |3-(-3)|) / 3 = (2 + 4 + 6) / 3 = 4.0
        mae = mean_absolute_error([1, -2, 3], [-1, 2, -3])
        assert mae == 4.0
    
    def test_float_values(self):
        """Test MAE with floating point values."""
        # MAE = (|1.5-1.0| + |2.5-2.0| + |3.5-3.0|) / 3 = (0.5 + 0.5 + 0.5) / 3 = 0.5
        mae = mean_absolute_error([1.5, 2.5, 3.5], [1.0, 2.0, 3.0])
        assert mae == 0.5
    
    def test_extreme_values(self):
        """Test MAE with extreme values."""
        # Very large numbers
        large_true = [1e6, 2e6, 3e6]
        large_pred = [1.1e6, 2.1e6, 3.1e6]
        # MAE = (|1e6-1.1e6| + |2e6-2.1e6| + |3e6-3.1e6|) / 3 = (1e5 + 1e5 + 1e5) / 3 = 1e5
        mae_large = mean_absolute_error(large_true, large_pred)
        assert mae_large == 1e5
        
        # Very small numbers
        small_true = [1e-6, 2e-6, 3e-6]
        small_pred = [1.1e-6, 2.1e-6, 3.1e-6]
        # MAE = (|1e-6-1.1e-6| + |2e-6-2.1e-6| + |3e-6-3.1e-6|) / 3 = (1e-7 + 1e-7 + 1e-7) / 3 = 1e-7
        mae_small = mean_absolute_error(small_true, small_pred)
        assert pytest.approx(mae_small, abs=1e-10) == 1e-7
    
    def test_different_input_types(self):
        """Test MAE with different input types that can be converted to numeric arrays."""
        # Lists
        assert mean_absolute_error([1, 2, 3], [2, 3, 4]) == 1.0
        
        # Numpy arrays
        assert mean_absolute_error(np.array([1, 2, 3]), np.array([2, 3, 4])) == 1.0
        
        # Mixed types
        assert mean_absolute_error([1, 2, 3], np.array([2, 3, 4])) == 1.0
        
        # Tuples
        assert mean_absolute_error((1, 2, 3), (2, 3, 4)) == 1.0
    
    def test_single_element_arrays(self):
        """Test MAE with single element arrays."""
        assert mean_absolute_error([5], [7]) == 2.0
    
    def test_input_validation_errors(self):
        """Test that input validation errors are properly propagated."""
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            mean_absolute_error([1, 2, 3], [1, 2])
        
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            mean_absolute_error([], [])
        
        with pytest.raises(TypeError, match="Inputs must be iterable"):
            mean_absolute_error(123, [1, 2, 3])


class TestPandasInputs:
    def test_pandas_series_inputs(self):
        """Test that pandas Series inputs are handled correctly."""
        # Create pandas Series
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([1.5, 2.5, 3.5])
        
        # Validate inputs
        y_true, y_pred = validate_inputs(s1, s2)
        
        # Check types and values
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        np.testing.assert_array_equal(y_true, np.array([1, 2, 3]))
        np.testing.assert_array_equal(y_pred, np.array([1.5, 2.5, 3.5]))
        
        # Test MAE calculation
        mae = mean_absolute_error(s1, s2)
        assert mae == 0.5
    
    def test_pandas_dataframe_inputs(self):
        """Test that pandas DataFrame inputs are handled correctly."""
        # Create pandas DataFrames
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        df2 = pd.DataFrame({'A': [1.5, 2.5, 3.5], 'B': [4.5, 5.5, 6.5]})
        
        # Validate inputs - this will flatten the DataFrames
        y_true, y_pred = validate_inputs(df1, df2)
        
        # Check that DataFrames were flattened correctly
        # Expected: [1, 4, 2, 5, 3, 6] and [1.5, 4.5, 2.5, 5.5, 3.5, 6.5]
        # or similar flattening pattern
        assert len(y_true) == 6
        assert len(y_pred) == 6
        
        # Test MAE calculation with DataFrames
        mae = mean_absolute_error(df1, df2)
        assert mae == 0.5
    
    def test_mixed_input_types(self):
        """Test that mixed input types are handled correctly."""
        # Create different input types
        array = np.array([1, 2, 3])
        series = pd.Series([1.5, 2.5, 3.5])
        
        # Test numpy array and pandas Series
        y_true, y_pred = validate_inputs(array, series)
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        np.testing.assert_array_equal(y_true, np.array([1, 2, 3]))
        np.testing.assert_array_equal(y_pred, np.array([1.5, 2.5, 3.5]))
        
        # Test list and pandas DataFrame
        list_data = [1, 2, 3]
        df = pd.DataFrame({'A': [1.5, 2.5, 3.5]})
        y_true, y_pred = validate_inputs(list_data, df)
        assert len(y_true) == 3
        assert len(y_pred) == 3
        
        # Test pandas Series and numpy array
        mae = mean_absolute_error(series, array)
        assert mae == 0.5
    
    def test_dataframe_with_single_column(self):
        """Test DataFrames with a single column."""
        df1 = pd.DataFrame({'A': [1, 2, 3]})
        df2 = pd.DataFrame({'A': [2, 3, 4]})
        
        mae = mean_absolute_error(df1, df2)
        assert mae == 1.0
    
    def test_dataframe_with_index(self):
        """Test DataFrames with custom indices."""
        df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
        df2 = pd.DataFrame({'A': [2, 3, 4]}, index=['a', 'b', 'c'])
        
        mae = mean_absolute_error(df1, df2)
        assert mae == 1.0
    
    def test_series_with_different_indices(self):
        """Test Series with different indices but same values."""
        s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        s2 = pd.Series([2, 3, 4], index=['x', 'y', 'z'])
        
        mae = mean_absolute_error(s1, s2)
        assert mae == 1.0
    
    def test_dataframe_column_subset(self):
        """Test using only specific columns from DataFrames."""
        df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        df2 = pd.DataFrame({'A': [2, 3, 4], 'B': [5, 6, 7], 'C': [8, 9, 10]})
        
        # Use only columns A and B
        mae = mean_absolute_error(df1[['A', 'B']], df2[['A', 'B']])
        assert mae == 1.0
