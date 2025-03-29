"""
Test of claude: no docstrings, no type hints, no manually coded
validation.
"""

import numpy as np

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))