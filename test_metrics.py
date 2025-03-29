import pytest
import numpy as np
from metrics import validate_inputs

def test_non_iterable_inputs():
    """Test that function fails when inputs are not iterable"""
    # Test with integer for y_true
    with pytest.raises(TypeError, match="Inputs must be iterable"):
        validate_inputs(42, [1, 2, 3])
    
    # Test with None for y_pred
    with pytest.raises(TypeError, match="Inputs must be iterable"):
        validate_inputs([1, 2, 3], None)
    
    # Test with both non-iterables
    with pytest.raises(TypeError, match="Inputs must be iterable"):
        validate_inputs(42, 3.14)

def test_non_convertible_inputs():
    """Test that function fails when inputs can't be converted to numeric arrays"""
    # Test with non-numeric elements
    with pytest.raises(ValueError, match="Input arrays must contain numeric values"):
        validate_inputs([1, 2, "three"], [1, 2, 3])
    
    # Test with mixed types that can't be converted properly
    with pytest.raises(ValueError, match="Input arrays must contain numeric values"):
        validate_inputs([1, 2, [3]], [1, 2, 3])

def test_mismatched_array_lengths():
    """Test that function fails when input arrays have different lengths"""
    with pytest.raises(ValueError, match="Input arrays must have the same length"):
        validate_inputs([1, 2, 3], [1, 2, 3, 4])

def test_empty_arrays():
    """Test that function fails when input arrays are empty"""
    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        validate_inputs([], [])

def test_arrays_with_nan_inf():
    """Test that function handles NaN and Inf values"""
    # The function doesn't explicitly check for NaN/Inf, so these should pass validation
    # but we should verify the behavior
    y_true = [1, 2, np.nan, 4]
    y_pred = [1, np.inf, 3, 4]
    
    # This should pass validation but might cause issues in calculations
    y_true_arr, y_pred_arr = validate_inputs(y_true, y_pred)
    assert np.isnan(y_true_arr[2])
    assert np.isinf(y_pred_arr[1])

def test_nested_iterables():
    """Test that function handles nested iterables"""
    # Nested lists that can't be properly flattened to 1D
    with pytest.raises(ValueError):
        validate_inputs([[1, 2], [3, 4]], [[1, 2], [3, 4]])
