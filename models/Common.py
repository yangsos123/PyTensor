"""
Some common functions for models
"""

import numpy as np


def toArray(L, f):
    """
    Transfer an integer or float to numpy array.
    """
    if (isinstance(f, float)):
        return np.ones((L)) * f
    elif (isinstance(f, int)):
        return np.ones((L), dtype=int) * f
    else:
        return f
