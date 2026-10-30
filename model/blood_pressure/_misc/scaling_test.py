import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

"""
Compare different scaling methods on a nonlinear dataset to decide for the most suitable one for scaling the hormone curves.
"""
def minmax_scale_value_by_array(x, ref_min=0, ref_max=1, lower_bound=0, upper_bound=1):
    """
    Scale data to [a, b] using a known physical min and max.
    Even if the observed data never reaches ref_min, the scaling
    keeps values > a.
    """
    scaled = lower_bound + (upper_bound - lower_bound) * (x - ref_min) / (ref_max - ref_min)
    return np.clip(scaled, lower_bound, upper_bound)

x = np.linspace(0, 10, 500)
y = np.sin(x) + 0.1*np.random.randn(500) + np.exp(x/5)  # nonlinear growth + noise

scalers = {
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
    "Robust": RobustScaler(),
    "Quantile": QuantileTransformer(output_distribution='uniform'),
}

own_implementation = [minmax_scale_value_by_array(val, min(y), max(y), lower_bound=0, upper_bound=1) for val in y]

fig, ax = plt.subplots()
ax.plot(x, y, label='Original')
ax.plot(x, own_implementation, label='Own MinMax')

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(y.reshape(-1,1)).flatten()
    # scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min()) # normalize to [0,1]
    ax.plot(x, scaled, label=name)

ax.legend()
plt.show()
