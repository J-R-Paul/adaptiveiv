import numpy as np
import pandas as pd
from typing import Union, Optional

class DataManager:
    def __init__(self):
        self._data = None
        self._data_type_flag = None
        self._split_indices = None
        self._columns = {
            'y': None,
            'w': None,
            'z': None,
            'group': None,
            'x': None
        }

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate that input is a pandas DataFrame"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

    def _handle_missing_values(self, df: pd.DataFrame, dropna: bool) -> pd.DataFrame:
        """Handle missing values in DataFrame"""
        if dropna:
            return df.dropna()
        elif df.isnull().values.any():
            raise ValueError("Input contains missing values and dropna=False")
        return df

    def _validate_columns(self, df: pd.DataFrame, y_col: str, w_col: str,
                         z_cols: Union[List[str], str], group_col: str,
                         x_cols: Optional[List[str]]) -> None:
        """Validate existence of specified columns in DataFrame"""
        def check_column(col: str, col_type: str) -> None:
            if col and col not in df.columns:
                raise ValueError(f"{col_type} column '{col}' not found in DataFrame")

        check_column(y_col, "Target (y_col)")
        check_column(w_col, "Weights (w_col)")
        check_column(group_col, "Group (group_col)")

        # Validate z_cols
        if isinstance(z_cols, str):
            check_column(z_cols, "Z (z_cols)")
        elif z_cols:
            for col in z_cols:
                check_column(col, "Z (z_cols)")

        # Validate x_cols
        if x_cols is not None:
            for col in x_cols:
                check_column(col, "Features (x_cols)")

    def _store_column_references(self, y_col: str, w_col: str,
                               z_cols: Union[List[str], str],
                               group_col: str,
                               x_cols: Optional[List[str]]) -> None:
        """Store column references in internal dictionary"""
        self._columns['y'] = y_col
        self._columns['w'] = w_col
        self._columns['z'] = z_cols
        self._columns['group'] = group_col
        self._columns['x'] = x_cols

    def _validate_arrays(self, y: np.ndarray, W: np.ndarray, Z: np.ndarray,
                        group: np.ndarray, X: Optional[np.ndarray]) -> None:
        """Validate input arrays for consistency"""
        n_samples = len(y)

        if len(W) != n_samples:
            raise ValueError("Weight array length must match target array length")

        if len(group) != n_samples:
            raise ValueError("Group array length must match target array length")

        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        elif Z.shape[0] != n_samples:
            raise ValueError("Z array rows must match target array length")

        if X is not None:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")
            if X.shape[0] != n_samples:
                raise ValueError("X array rows must match target array length")

    def _create_data_dict(self, y: np.ndarray, W: np.ndarray, Z: np.ndarray,
                         group: np.ndarray, X: Optional[np.ndarray],
                         x_cols: Optional[List[str]], kwargs: dict) -> dict:
        """Create dictionary for DataFrame construction"""
        data_dict = {
            kwargs.get('y_col', 'y'): y,
            kwargs.get('w_col', 'w'): W,
            kwargs.get('group_col', 'group'): group
        }

        # Handle Z columns
        z_cols = kwargs.get('z_cols', self._generate_z_cols(Z))
        if isinstance(z_cols, str):
            data_dict[z_cols] = Z.ravel() if Z.ndim == 2 and Z.shape[1] == 1 else Z
        else:
            for i, col in enumerate(z_cols):
                data_dict[col] = Z[:, i]

        # Handle X columns if present
        if X is not None:
            x_names = x_cols or [f'x_{i}' for i in range(X.shape[1])]
            for i, col in enumerate(x_names):
                data_dict[col] = X[:, i]

        return data_dict

    def _generate_z_cols(self, Z: np.ndarray) -> List[str]:
        """Generate default column names for Z array"""
        return [f'z_{i}' for i in range(Z.shape[1])]

    @property
    def data(self) -> pd.DataFrame:
        """Return the loaded data"""
        if self._data is None:
            raise ValueError("No data loaded")
        return self._data.copy()

    @property
    def split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the split indices"""
        if self._split_indices is None:
            raise ValueError("No split indices available. Call split_data_in_two() first")
        return self._split_indices

    @property
    def columns(self) -> dict:
        """Return column mappings"""
        return self._columns.copy()
