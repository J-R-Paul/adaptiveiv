import numpy as np
import pandas as pd
from adaptiveiv import AdaptiveIV

model = AdaptiveIV()

# Load data from a pandas DataFrame
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'target': [1, 2, 3, 4, 5],
        'weight': [0.5, 1.0, 1.0, 0.8, 1.0],
        'z1': [10, 20, 30, 40, 50],
        'z2': [15, 25, 35, 45, 55],
        'group': [1, 1, 2, 2, 3],
        'x1': [100, 200, 300, 400, 500],
        'x2': [150, 250, 350, 450, 550]
    })


def sample_arrays():
    """Create sample arrays for testing"""
    y = np.array([1, 2, 3, 4, 5])
    W = np.array([0.5, 1.0, 1.0, 0.8, 1.0])
    Z = np.array([[10, 15], [20, 25], [30, 35], [40, 45], [50, 55]])
    group = np.array([1, 1, 2, 2, 3])
    X = np.array([[100, 150], [200, 250], [300, 350], [400, 450], [500, 550]])
    return y, W, Z, group, X


(model
 .load_data(data=sample_df(), y_col='target', w_col='weight', z_cols=['z1', 'z2'], group_col='group')
)

print(model.data_manager._data)

# Load data from numpy arrays
y, W, Z, group, X = sample_arrays()
(model
    .load_data(y=y, W=W, Z=Z, group=group)
)

print(model.data_manager._data)
