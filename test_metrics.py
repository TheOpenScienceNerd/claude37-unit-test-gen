import pytest
import numpy as np
import pandas as pd
import warnings

import metrics as m

class TestValidateInputs:
    """Tests for the validate_inputs function"""
    
    def test_valid_inputs(self):
        """Test with valid inputs of different types"""
        # Lists
        y_true, y_pred = m.validate_inputs([1, 2, 3], [4, 5, 6])
        assert np.array_equal(y_true, np.array([1, 2, 3]))
        assert np.array_equal(y_pred, np.array([4, 5, 6]))
        
        # NumPy arrays
        y_true, y_pred = m.validate_inputs(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert np.array_equal(y_true, np.array([1, 2, 3]))
        assert np.array_equal(y_pred, np.array([4, 5, 6]))
        
        # Pandas Series
        y_true, y_pred = m.validate_inputs(pd.Series([1, 2, 3]), pd.Series([4, 5, 6]))
        assert np.array_equal(y_true, np.array([1, 2, 3]))
        assert np.array_equal(y_pred, np.array([4, 5, 6]))
        
        # Pandas DataFrames
        y_true, y_pred = m.validate_inputs(pd.DataFrame([1, 2, 3]), pd.DataFrame([4, 5, 6]))
        assert np.array_equal(y_true, np.array([1, 2, 3]))
        assert np.array_equal(y_pred, np.array([4, 5, 6]))
        
        # 2D arrays
        y_true, y_pred = m.validate_inputs(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
        assert np.array_equal(y_true, np.array([1, 2, 3, 4]))
        assert np.array_equal(y_pred, np.array([5, 6, 7, 8]))
    
    def test_mixed_types(self):
        """Test with mixed input types"""
        y_true, y_pred = m.validate_inputs([1, 2, 3], pd.Series([4, 5, 6]))
        assert np.array_equal(y_true, np.array([1, 2, 3]))
        assert np.array_equal(y_pred, np.array([4, 5, 6]))
        
        y_true, y_pred = m.validate_inputs(np.array([1, 2, 3]), [4.0, 5.0, 6.0])
        assert np.array_equal(y_true, np.array([1, 2, 3]))
        assert np.array_equal(y_pred, np.array([4.0, 5.0, 6.0]))
    
    def test_error_different_lengths(self):
        """Test error when inputs have different lengths"""
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            m.validate_inputs([1, 2, 3], [4, 5])
    
    def test_error_empty_arrays(self):
        """Test error when inputs are empty"""
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            m.validate_inputs([], [])
    
    def test_error_non_numeric(self):
        """Test error when inputs contain non-numeric values"""
        with pytest.raises(ValueError, match="Input arrays must contain numeric values"):
            m.validate_inputs(['a', 'b', 'c'], [1, 2, 3])
    
    def test_error_non_iterable(self):
        """Test error when inputs are not iterable"""
        with pytest.raises(TypeError, match="Inputs must be iterable"):
            m.validate_inputs(123, [1, 2, 3])
        
        with pytest.raises(TypeError, match="Inputs must be iterable"):
            m.validate_inputs([1, 2, 3], None)


class TestMeanAbsoluteError:
    """Tests for the mean_absolute_error function"""
    
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            # Basic test cases with lists
            ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0.0),
            ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], 6.0),
            ([103, 130, 132, 124, 124, 108], [129, 111, 122, 129, 110, 141], 17.833333),
            
            # Different array-like data types
            (np.array([1, 2, 3]), np.array([4, 5, 6]), 3.0),
            (pd.Series([1, 2, 3]), [4, 5, 6], 3.0),
            ([1, 2, 3], pd.Series([4, 5, 6]), 3.0),
            (pd.DataFrame([1, 2, 3]), pd.DataFrame([4, 5, 6]), 3.0),
            (pd.DataFrame([1, 2, 3]), pd.Series([4, 5, 6]), 3.0),
            (pd.DataFrame([1, 2, 3]), np.array([4, 5, 6]), 3.0),
            
            # Float values
            ([1.5, 2.5, 3.5], [1.0, 2.0, 3.0], 0.5),
            
            # Mixed types
            ([1, 2, 3], [1.5, 2.5, 3.5], 0.5),
            
            # 2D arrays (should be flattened)
            (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), 4.0),
            
            # Edge cases
            ([0, 0, 0], [0, 0, 0], 0.0),
            ([-1, -2, -3], [-1, -2, -3], 0.0),
            ([-1, -2, -3], [1, 2, 3], 4.0),
            
            # Extreme values
            ([1e6, 2e6, 3e6], [1e6, 2e6, 3e6], 0.0),
            ([1e-6, 2e-6, 3e-6], [2e-6, 3e-6, 4e-6], 1e-6),
        ]
    )
    def test_mean_absolute_error(self, y_true, y_pred, expected):
        """Test mean absolute error calculation with various input types"""
        error = m.mean_absolute_error(y_true, y_pred)
        assert pytest.approx(expected, rel=1e-5) == error
    
    def test_error_propagation(self):
        """Test that errors from validate_inputs are properly propagated"""
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            m.mean_absolute_error([1, 2, 3], [4, 5])
        
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            m.mean_absolute_error([], [])
        
        with pytest.raises(TypeError, match="Inputs must be iterable"):
            m.mean_absolute_error(123, [1, 2, 3])
    
    def test_with_nan_values(self):
        """Test behavior with NaN values (should propagate to result)"""
        # This test demonstrates current behavior, which returns NaN when NaN values are present
        result = m.mean_absolute_error([1, 2, np.nan], [4, 5, 6])
        assert np.isnan(result)
    
    def test_with_inf_values(self):
        """Test behavior with infinity values"""
        # This test demonstrates current behavior with infinity values
        result = m.mean_absolute_error([1, 2, np.inf], [4, 5, 6])
        assert np.isinf(result)

class TestMeanAbsolutePercentageError:
    """Tests for the mean_absolute_percentage_error function"""
    
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            # Basic test cases
            ([100, 200, 300], [90, 210, 310], 6.666667),
            ([10, 20, 30, 40, 50], [11, 22, 33, 44, 55], 10.0),
            ([1000, 1000, 1000], [900, 1100, 1000], 6.666667),
            
            # Different magnitudes
            ([1, 10, 100], [1.1, 11, 110], 10.0),
            ([0.1, 0.2, 0.3], [0.11, 0.22, 0.33], 10.0),
            
            # Negative values
            ([-100, -200, -300], [-90, -210, -310], 6.666667),
            ([-10, -20, -30], [-11, -22, -33], 10.0),
            
            # Mixed positive and negative values
            ([10, -20, 30], [11, -22, 33], 10.0),
            
            # Underforecasting and overforecasting
            ([100, 100, 100], [90, 90, 90], 10.0),  # Underforecasting
            ([100, 100, 100], [110, 110, 110], 10.0),  # Overforecasting
        ]
    )
    def test_mape_calculation(self, y_true, y_pred, expected):
        """Test MAPE calculation with various numeric inputs"""
        error = m.mean_absolute_percentage_error(y_true, y_pred)
        assert pytest.approx(expected, rel=1e-5) == error
    
    @pytest.mark.parametrize(
        "y_true, y_pred, expected",
        [
            # Different array-like data types
            (np.array([100, 200, 300]), np.array([90, 210, 310]), 6.666667),
            (pd.Series([10, 20, 30]), pd.Series([11, 22, 33]), 10.0),
            (pd.DataFrame([100, 200, 300]), np.array([90, 210, 310]), 6.666667),
            
            # 2D arrays (should be flattened)
            (np.array([[100, 200], [300, 400]]), np.array([[90, 210], [310, 410]]), 5.0),
        ]
    )
    def test_mape_data_types(self, y_true, y_pred, expected):
        """Test MAPE calculation with various input data types"""
        error = m.mean_absolute_percentage_error(y_true, y_pred)
        assert pytest.approx(expected, rel=1e-5) == error
    
    def test_mape_near_zero_warning(self):
        """Test that a warning is issued for values close to zero"""
        with pytest.warns(UserWarning, match="very close to zero"):
            m.mean_absolute_percentage_error([1e-11, 1, 2], [2e-11, 1.1, 2.2])
    
    def test_mape_asymmetry(self):
        """Test the asymmetric property of MAPE"""
        # Demonstrate that underforecasting and overforecasting by the same amount
        # produce different MAPE values when true values differ
        y_true = [10, 100, 1000]
        
        # Underforecast by 10%
        y_under = [9, 90, 900]
        under_mape = m.mean_absolute_percentage_error(y_true, y_under)
        
        # Overforecast by 10%
        y_over = [11, 110, 1100]
        over_mape = m.mean_absolute_percentage_error(y_true, y_over)
        
        # Both should be 10%, demonstrating symmetry for consistent percentage errors
        assert pytest.approx(10.0) == under_mape
        assert pytest.approx(10.0) == over_mape
        
        # But with inconsistent errors, asymmetry appears
        y_mixed = [11, 90, 1100]  # +10%, -10%, +10%
        mixed_mape = m.mean_absolute_percentage_error(y_true, y_mixed)
        assert mixed_mape != under_mape  # Should not be equal


class TestMeanAbsolutePercentageErrorEdgeCases:
    """Tests for edge cases and error handling in mean_absolute_percentage_error"""
    
    def test_error_with_zeros(self):
        """Test that an error is raised when y_true contains zeros"""
        with pytest.raises(ValueError, match="cannot be calculated when actual values.*contain zeros"):
            m.mean_absolute_percentage_error([0, 1, 2], [1, 2, 3])
        
        with pytest.raises(ValueError, match="cannot be calculated when actual values.*contain zeros"):
            m.mean_absolute_percentage_error([1, 0, 3], [1, 1, 3])
    
    def test_error_propagation(self):
        """Test that errors from validate_inputs are properly propagated"""
        with pytest.raises(ValueError, match="Input arrays must have the same length"):
            m.mean_absolute_percentage_error([1, 2, 3], [4, 5])
        
        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            m.mean_absolute_percentage_error([], [])
        
        with pytest.raises(TypeError, match="Inputs must be iterable"):
            m.mean_absolute_percentage_error(123, [1, 2, 3])
        
        with pytest.raises(ValueError, match="Input arrays must contain numeric values"):
            m.mean_absolute_percentage_error(['a', 'b', 'c'], [1, 2, 3])
    
    def test_with_very_small_values(self):
        """Test behavior with very small but non-zero values"""
        # This should issue a warning but still calculate
        with pytest.warns(UserWarning, match="very close to zero"):
            result = m.mean_absolute_percentage_error([1e-11, 1e-11], [2e-11, 3e-11])
            # Result should be very large due to the relative error
            assert result > 100  # MAPE will be 100% or higher
    
    def test_with_nan_values(self):
        """Test behavior with NaN values"""
        # NaN values should propagate through calculation
        result = m.mean_absolute_percentage_error([1, 2, np.nan], [1.1, 2.2, 3.3])
        assert np.isnan(result)
    
    def test_with_inf_values(self):
        """Test behavior with infinity values"""
        # Infinity values should propagate through calculation
        result = m.mean_absolute_percentage_error([1, 2, np.inf], [1.1, 2.2, 3.3])
        assert np.isinf(result)
    
    def test_extreme_values(self):
        """Test with extreme values to check numerical stability"""
        # Very large values
        large_result = m.mean_absolute_percentage_error([1e10, 2e10], [1.1e10, 2.2e10])
        assert pytest.approx(10.0) == large_result
        
        # Very small but valid values
        with pytest.warns(UserWarning):
            small_result = m.mean_absolute_percentage_error([1e-5, 2e-5], [1.1e-5, 2.2e-5])
            assert pytest.approx(10.0) == small_result
