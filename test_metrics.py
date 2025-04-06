import pytest
import numpy as np
import pandas as pd
from metrics import mean_absolute_error

# Test data as tuples of (y_true, y_pred, expected_mae)
test_cases = [
    ([3, 5, 2, 8], [2, 5, 4, 10], 1.25),
    ([7, 3, 6, 1, 9], [5, 2, 8, 1, 10], 1.2),
    ([10, 20, 30, 40, 50], [12, 18, 28, 42, 48], 2.0),
    ([4, 8, 15, 16, 23, 42], [5, 7, 14, 17, 22, 40], 1.166666667),
    ([100, 200, 300, 400, 500, 600, 700], [110, 190, 290, 410, 490, 590, 710], 10.0)
]

class TestMeanAbsoluteError:
    """Test suite for the mean_absolute_error function."""
    
    @pytest.mark.parametrize("y_true, y_pred, expected", test_cases)
    def test_list_inputs(self, y_true, y_pred, expected):
        """Test MAE calculation with list inputs."""
        result = mean_absolute_error(y_true, y_pred)
        assert result == pytest.approx(expected)
    
    @pytest.mark.parametrize("y_true, y_pred, expected", test_cases)
    def test_numpy_array_inputs(self, y_true, y_pred, expected):
        """Test MAE calculation with numpy array inputs."""
        result = mean_absolute_error(np.array(y_true), np.array(y_pred))
        assert result == pytest.approx(expected)
    
    @pytest.mark.parametrize("y_true, y_pred, expected", test_cases)
    def test_pandas_series_inputs(self, y_true, y_pred, expected):
        """Test MAE calculation with pandas Series inputs."""
        result = mean_absolute_error(pd.Series(y_true), pd.Series(y_pred))
        assert result == pytest.approx(expected)
    
    @pytest.mark.parametrize("y_true, y_pred, expected", test_cases)
    def test_mixed_inputs(self, y_true, y_pred, expected):
        """Test MAE calculation with mixed input types."""
        result = mean_absolute_error(y_true, np.array(y_pred))
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("y_true, y_pred, expected", test_cases)
    def test_reversed_mixed_inputs(self, y_true, y_pred, expected):
        """Test MAE calculation with reversed mixed input types."""
        result = mean_absolute_error(np.array(y_true), pd.Series(y_pred))
        assert result == pytest.approx(expected)
