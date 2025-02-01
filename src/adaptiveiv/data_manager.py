import numpy as np
import pandas as pd
from typing import Union, Optional, List, Tuple

class DataManager:
    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._columns: dict = {}
        self._split_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def load_data(self,
                  df: Optional[pd.DataFrame] = None,
                  y: Optional[np.ndarray] = None,
                  W: Optional[np.ndarray] = None,
                  Z: Optional[np.ndarray] = None,
                  group: Optional[np.ndarray] = None,
                  X: Optional[np.ndarray] = None,
                  *,
                  y_col: str = 'y',
                  w_col: str = 'w',
                  z_cols: Union[str, List[str]] = 'z',
                  group_col: str = 'group',
                  x_cols: Optional[List[str]] = None,
                  dropna: bool = False) -> None:
        """
        Load data from either a pandas DataFrame or numpy arrays.
        """
        if df is not None:
            self._load_from_dataframe(df, y_col, w_col, z_cols, group_col, x_cols, dropna)
        elif all(arg is not None for arg in (y, W, Z, group)):
            self._load_from_arrays(y, W, Z, group, X, x_cols, y_col, w_col, z_cols, group_col, dropna)
        else:
            raise ValueError("Either a pandas DataFrame or the required numpy arrays must be provided.")

    def _load_from_dataframe(self,
                             df: pd.DataFrame,
                             y_col: str,
                             w_col: str,
                             z_cols: Union[str, List[str]],
                             group_col: str,
                             x_cols: Optional[List[str]],
                             dropna: bool) -> None:
        """Load and validate data from a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Provided data is not a pandas DataFrame.")

        if dropna:
            df = df.dropna()
        elif df.isnull().values.any():
            raise ValueError("Data contains missing values; set dropna=True to remove them.")

        self._data = df.copy()
        self._columns = {'y': y_col, 'w': w_col, 'z': z_cols, 'group': group_col, 'x': x_cols}
        self._split_indices = None

    def _load_from_arrays(self,
                          y: np.ndarray,
                          W: np.ndarray,
                          Z: np.ndarray,
                          group: np.ndarray,
                          X: Optional[np.ndarray],
                          x_cols: Optional[List[str]],
                          y_col: str,
                          w_col: str,
                          z_cols: Union[str, List[str]],
                          group_col: str,
                          dropna: bool) -> None:
        """Load and validate data from numpy arrays, then convert to a pandas DataFrame."""
        n_samples = len(y)
        if len(W) != n_samples or len(group) != n_samples:
            raise ValueError("Length of W and group must match length of y.")
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if Z.shape[0] != n_samples:
            raise ValueError("Number of rows in Z must match length of y.")

        df = pd.DataFrame({y_col: y, w_col: W, group_col: group})
        if X is not None:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != n_samples:
                raise ValueError("Number of rows in X must match length of y.")
            x_col_names = x_cols if x_cols is not None else [f'X{i}' for i in range(X.shape[1])]
            df = pd.concat([df, pd.DataFrame(X, columns=x_col_names)], axis=1)

        self._load_from_dataframe(df, y_col, w_col, z_cols, group_col, x_cols, dropna)

    def split_data(self, prop: float = 0.5, random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into two subsets.
        """
        if self._data is None:
            raise ValueError("No data loaded.")

        rng = np.random.RandomState(random_state)
        indices = rng.permutation(len(self._data))
        split_point = int(len(indices) * prop)
        self._split_indices = (indices[:split_point], indices[split_point:])
        return (self._data.iloc[self._split_indices[0]].copy(),
                self._data.iloc[self._split_indices[1]].copy())

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("No data loaded.")
        return self._data.copy()

    @property
    def columns(self) -> dict:
        return self._columns.copy()

    @property
    def split_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._split_indices is None:
            raise ValueError("No split indices available. Please run split_data() first.")
        return self._split_indices

# # Example usage for testing
# if __name__ == "__main__":
#     df = pd.DataFrame({
#         'y': np.random.randn(100),
#         'w': np.random.randn(100),
#         'z': np.random.randn(100),
#         'group': np.random.randint(0, 5, 100),
#         'X1': np.random.randn(100),
#         'X2': np.random.randn(100),
#         'X3': np.random.randn(100)
#     })
#     manager = DataManager()
#     manager.load_data(df=df, y_col='y', w_col='w', z_cols='z', group_col='group', x_cols=['X1', 'X2', 'X3'])
#     print("Loaded Data from DataFrame:")
#     print(manager._data.head())

#     y = df['y'].values
#     w = df['w'].values
#     z = df['z'].values
#     group = df['group'].values
#     x = df[['X1', 'X2', 'X3']].values

#     manager.load_data(y=y, W=w, Z=z, group=group, X=x, x_cols=['X1', 'X2', 'X3'])
#     print("Loaded Data from Arrays:")
#     print(manager._data.head())

#     manager.load_data(y=y, W=w, Z=z, group=group)
#     print("Loaded Data from Arrays without X:")
#     print(manager._data.head())


#     d1, d2 = manager.split_data()
#     print(d1.head())
#     print(d2.head())

#     print(manager.columns)
