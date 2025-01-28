import numpy as np
import pandas as pd
from typing import Union, Optional

class DataManager:
    def __init__(self):
        self._data_type_flag = None

    def load_pandas(self,
                    df: pd.DataFrame,
                    y_col: str = "",
                    w_col: str = "",
                    z_cols: Union[list, str] = "",
                    group_col: str = "",
                    x_cols: Union[str, list, None] = None,
                    dropna: bool = False) -> None:
        """
        Load data from pandas DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        y_col : str, optional
            Name of target column
        w_col : str, optional
            Name of endogenous column
        z_cols : Union[list, str], optional
            Name(s) of exoegnous instruments
        group_col : str, optional
            Name of grouping column
        x_cols : Union[str, list, None], optional
            Names of controls (None for all non-special columns)
        dropna : bool, default False
            Whether to drop rows with missing values
        """
        # Input validation for DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Handle missing values
        if dropna:
            df = df.dropna()
        elif df.isnull().values.any():
            raise ValueError("Input contains missing values and dropna=False")

        # Validate column existence
        def validate_column(name: str, col_type: str):
            if name and name not in df.columns:
                raise ValueError(f"{col_type} column '{name}' not found in DataFrame")

        # Validate y_col
        validate_column(y_col, "Target (y_col)")

        # Validate w_col
        validate_column(w_col, "Weights (w_col)")

        # Validate z_cols
        if z_cols:
            if isinstance(z_cols, str):
                validate_column(z_cols, "Z (z_cols)")
            else:
                for col in z_cols:
                    validate_column(col, "Z (z_cols)")

        # Validate group_col
        validate_column(group_col, "Group (group_col)")

        # Validate x_cols
        if x_cols is not None:
            if isinstance(x_cols, str):
                validate_column(x_cols, "Features (x_cols)")
            else:
                for col in x_cols:
                    validate_column(col, "Features (x_cols)")

        # Store validated data
        self._data = df
        self._y_col = y_col
        self._w_col = w_col
        self._z_cols = z_cols
        self._group_col = group_col
        self._x_cols = x_cols
        self._target = y_col
        self._data_type_flag = "pandas"

    def load_from_arrays(self,
                        y: np.ndarray,
                        W: np.ndarray,
                        Z: np.ndarray,
                        group: np.ndarray,
                        X: Optional[np.ndarray] = None,
                        y_col: str = "y",
                        w_col: str = "w",
                        z_cols: Union[str, list] = "z",
                        group_col: str = "group",
                        x_cols: Optional[list] = None,
                        dropna: bool = False) -> None:
        """
        Load data from numpy arrays into DataFrame format

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : Optional[np.ndarray]
            Target vector (n_samples,)
        w : Optional[np.ndarray]
            Weights vector (n_samples,)
        z : Optional[np.ndarray]
            Additional features (n_samples, n_z_features)
        group : Optional[np.ndarray]
            Group identifiers (n_samples,)
        x_cols : Optional[list]
            Names for feature columns
        y_col : str, default "target"
            Name for target column
        w_col : str, default "weights"
            Name for weights column
        z_cols : Optional[list]
            Names for additional feature columns
        group_col : str, default "group"
            Name for group column
        dropna : bool, default False
            Whether to drop rows with missing values
        """
        data_dict = {}

        n_y = len(y)
        data_dict[y_col] = n_y

        # Handle control matrix
        if X is not None:
            if X.ndim != 2:
                raise ValueError("X must be a 2D array")

            if X.shape[0] != n_y:
                raise ValueError("X row count must match y length")

            n_features = X.shape[1]
            x_cols = x_cols or [f"x_{i}" for i in range(n_features)]
            if len(x_cols) != n_features:
                raise ValueError("Length of x_cols must match number of features in X")

            for i, col in enumerate(x_cols):
                data_dict[col] = X[:, i]


        # Handle weights

            if len(W) != n_y:
                raise ValueError("weights length must match y length")
            data_dict[w_col] = W

        # Handle additional features (z)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        n_z_features = Z.shape[1]
        z_cols = z_cols or [f"z_{i}" for i in range(n_z_features)]
        if len(z_cols) != n_z_features:
            raise ValueError("Length of z_cols must match number of features in z")

        for i, col in enumerate(z_cols):
            data_dict[col] = Z[:, i]

        # Handle groups
        if len(group) != n_y:
            raise ValueError("group length must match y length")
        data_dict[group_col] = group

        # Create DataFrame and load
        df = pd.DataFrame(data_dict)
        self.load_pandas(
            df=df,
            y_col=y_col,
            w_col=w_col,
            z_cols=z_cols,
            group_col=group_col,
            x_cols=x_cols,
            dropna=dropna
        )

    def split_data_in_two(
        self,
        prop: float = 0.5,
        random_state: Optional[int] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the currently loaded DataFrame into two random subsets.

        Parameters
        ----------
        prop : float, optional
            Proportion of rows to include in the first subset. Must be between 0 and 1.
            Defaults to 0.5 (i.e., split in half).
        random_state : int, optional
            Seed for the random number generator, to make the split reproducible.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            Two new DataFrames: first subset has prop * total_rows, and the second has
            the remainder of the rows.
        """
        # Check that data is loaded and stored as a pandas DataFrame
        if self._data_type_flag != "pandas" or not hasattr(self, "_data"):
            raise ValueError("No data loaded, or data is not a pandas DataFrame.")

        if not (0.0 <= prop <= 1.0):
            raise ValueError("prop must be between 0 and 1.")

        df_length = len(self._data)
        # Create a random number generator
        rng = np.random.default_rng(random_state)
        # Create and shuffle an array of indices
        indices = np.arange(df_length)
        rng.shuffle(indices)

        # Determine where to split the array
        split_ind = int(df_length * prop)

        # Split indices into two halves
        split_indices_1 = indices[:split_ind]
        split_indices_2 = indices[split_ind:]

        # Extract DataFrame subsets
        df_first_half = self._data.iloc[split_indices_1].copy()
        df_second_half = self._data.iloc[split_indices_2].copy()

        # Store split indices for reference
        self._split_indices_1 = split_indices_1
        self._split_indices_2 = split_indices_2

        return df_first_half, df_second_half


    def data(self):
        """
        Return the loaded data as a pandas DataFrame

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        if self._data_type_flag != "pandas" or not hasattr(self, "_data"):
            raise ValueError("No data loaded, or data is not a pandas DataFrame.")

        return self._data

    def split_indicies(self):
        """
        Return the split indices

        Returns
        -------
        tuple
            Tuple of split indices
        """
        if not hasattr(self, "_split_indices_1") or not hasattr(self, "_split_indices_2"):
            raise ValueError("No split indices stored. Call split_data_in_two() first.")

        return self._split_indices_1, self._split_indices_2
