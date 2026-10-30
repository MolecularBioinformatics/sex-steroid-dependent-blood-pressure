import numpy as np

def minmax_scale_value_by_array(x, ref_min=0, ref_max=1, lower_bound=0, upper_bound=1):
    """
    Scale data to [a, b] using a known physical min and max.
    Even if the observed data never reaches ref_min, the scaling
    keeps values > a.
    """
    scaled = lower_bound + (upper_bound - lower_bound) * (x - ref_min) / (ref_max - ref_min)
    return np.clip(scaled, lower_bound, upper_bound)

def hill_pos(x, n, S=0.5):
    return ((x/S)**n)/(1+(x/S)**n)

def hill_neg(x, n, S=0.5):
    return S**n/(S**n + x**n)
