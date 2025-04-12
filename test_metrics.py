import pytest
import numpy as np
import pandas as pd
from metrics import mean_absolute_error
from metrics import _validate_single_array

@pytest.mark.parametrize("y_true, y_pred, expected", [
    ([3, 5, 2, 8], [2, 5, 4, 10], 1.25),
    ([7, 3, 6, 1, 9], [5, 2, 8, 1, 10], 1.2),
    ([10, 20, 30, 40, 50], [12, 18, 28, 42, 48], 2.0),
    ([4, 8, 15, 16, 23, 42], [5, 7, 14, 17, 22, 40], 1.166666667),
    ([100, 200, 300, 400, 500, 600, 700], [110, 190, 290, 410, 490, 590, 710], 10.0),
])
def test_mean_absolute_error(y_true, y_pred, expected):
    result = mean_absolute_error(y_true, y_pred)
    assert result == pytest.approx(expected)

def test_mean_absolute_error_scalar():
    assert mean_absolute_error(3, 2) == 1.0

def test_mean_absolute_error_numpy_arrays():
    y_true = np.array([3, 5, 2, 8])
    y_pred = np.array([2, 5, 4, 10])
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(1.25)

def test_mean_absolute_error_mixed_types():
    y_true = [3, 5, 2, 8]
    y_pred = np.array([2, 5, 4, 10])
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(1.25)

def test_empty_array():
    """Test that empty arrays raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_single_array(np.array([]), "test_array")

def test_array_with_nan():
    """Test that arrays with NaN values raise ValueError."""
    with pytest.raises(ValueError, match="contains NaN values"):
        _validate_single_array(np.array([1.0, np.nan, 3.0]), "test_array")

def test_array_with_infinity():
    """Test that arrays with infinity values raise ValueError."""
    with pytest.raises(ValueError, match="contains infinity values"):
        _validate_single_array(np.array([1.0, np.inf, 3.0]), "test_array")
        
    with pytest.raises(ValueError, match="contains infinity values"):
        _validate_single_array(np.array([1.0, -np.inf, 3.0]), "test_array")

def test_boolean_array():
    """Test that boolean arrays raise ValueError."""
    with pytest.raises(ValueError, match="contains boolean values"):
        _validate_single_array(np.array([True, False, True]), "test_array")

def test_zero_array_warning():
    """Test that arrays with all zeros generate a warning."""
    with pytest.warns(UserWarning, match="All values in test_array are zero"):
        result = _validate_single_array(np.array([0, 0, 0]), "test_array")
        assert np.array_equal(result, np.array([0, 0, 0]))

def test_very_large_values():
    """Test with very large values that might cause numerical issues."""
    y_true = [1e15, 2e15, 3e15]
    y_pred = [1.1e15, 2.2e15, 3.3e15]
    # Expected: (0.1e15 + 0.2e15 + 0.3e15) / 3 = 0.2e15
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2e14)

def test_very_small_values():
    """Test with very small values that might cause precision issues."""
    y_true = [1e-10, 2e-10, 3e-10]
    y_pred = [1.1e-10, 2.2e-10, 3.3e-10]
    # Expected: (0.1e-10 + 0.2e-10 + 0.3e-10) / 3 = 0.2e-10
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2e-11)

def test_mixed_large_and_small():
    """Test with mixed very large and very small values."""
    y_true = [1e10, 1e-10]
    y_pred = [2e10, 2e-10]
    # Expected: (1e10 + 1e-10) / 2 ≈ 5e9
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(5e9)

def test_opposite_signs():
    """Test with values of opposite signs."""
    y_true = [100, -100, 50, -50]
    y_pred = [-100, 100, -50, 50]
    # Expected: (200 + 200 + 100 + 100) / 4 = 150
    assert mean_absolute_error(y_true, y_pred) == 150.0

def test_different_input_types():
    """Test with different input types."""
    # DataFrame
    df_true = pd.DataFrame({'value': [3, 5, 2, 8]})
    df_pred = pd.DataFrame({'value': [2, 5, 4, 10]})
    assert mean_absolute_error(df_true, df_pred) == pytest.approx(1.25)
    
    # Series
    series_true = pd.Series([3, 5, 2, 8])
    series_pred = pd.Series([2, 5, 4, 10])
    assert mean_absolute_error(series_true, series_pred) == pytest.approx(1.25)
    
    # Mixed types
    assert mean_absolute_error(df_true, series_pred) == pytest.approx(1.25)

def test_identical_values():
    """Test with identical true and predicted values."""
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1, 2, 3, 4, 5]
    assert mean_absolute_error(y_true, y_pred) == 0.0

def test_single_value():
    """Test with single value arrays."""
    assert mean_absolute_error([10], [12]) == 2.0





