import pytest
import numpy as np
import pandas as pd
from typing import List, Tuple

from adaptiveiv.core.data_manager import DataManager

@pytest.fixture
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

@pytest.fixture
def sample_arrays() -> Tuple[np.ndarray, ...]:
    """Create sample arrays for testing"""
    y = np.array([1, 2, 3, 4, 5])
    W = np.array([0.5, 1.0, 1.0, 0.8, 1.0])
    Z = np.array([[10, 15], [20, 25], [30, 35], [40, 45], [50, 55]])
    group = np.array([1, 1, 2, 2, 3])
    X = np.array([[100, 150], [200, 250], [300, 350], [400, 450], [500, 550]])
    return y, W, Z, group, X

class TestDataManager:
    def test_init(self):
        """Test initialization of DataManager"""
        dm = DataManager()
        assert dm._data is None
        assert dm._data_type_flag is None
        assert dm._split_indices is None
        assert all(v is None for v in dm._columns.values())

    def test_load_pandas_valid(self, sample_df):
        """Test loading valid pandas DataFrame"""
        dm = DataManager()
        dm.load_pandas(
            df=sample_df,
            y_col='target',
            w_col='weight',
            z_cols=['z1', 'z2'],
            group_col='group',
            x_cols=['x1', 'x2']
        )
        assert dm._data_type_flag == "pandas"
        assert dm._data.equals(sample_df)
        assert dm._columns['y'] == 'target'
        assert dm._columns['w'] == 'weight'
        assert dm._columns['z'] == ['z1', 'z2']
        assert dm._columns['group'] == 'group'
        assert dm._columns['x'] == ['x1', 'x2']

    def test_load_pandas_invalid_df(self):
        """Test loading invalid DataFrame"""
        dm = DataManager()
        with pytest.raises(TypeError):
            dm.load_pandas(
                df=[1, 2, 3],  # Not a DataFrame
                y_col='target',
                w_col='weight',
                z_cols=['z1'],
                group_col='group'
            )

    def test_load_pandas_missing_columns(self, sample_df):
        """Test loading DataFrame with missing columns"""
        dm = DataManager()
        with pytest.raises(ValueError):
            dm.load_pandas(
                df=sample_df,
                y_col='nonexistent',
                w_col='weight',
                z_cols=['z1'],
                group_col='group'
            )

    def test_load_pandas_with_na(self):
        """Test loading DataFrame with NA values"""
        df = pd.DataFrame({
            'target': [1, 2, np.nan, 4],
            'weight': [0.5, 1.0, 1.0, 0.8],
            'z1': [10, 20, 30, 40],
            'group': [1, 1, 2, 2]
        })

        dm = DataManager()
        # Should raise error when dropna=False
        with pytest.raises(ValueError):
            dm.load_pandas(
                df=df,
                y_col='target',
                w_col='weight',
                z_cols=['z1'],
                group_col='group',
                dropna=False
            )

        # Should work when dropna=True
        dm.load_pandas(
            df=df,
            y_col='target',
            w_col='weight',
            z_cols=['z1'],
            group_col='group',
            dropna=True
        )
        assert len(dm._data) == 3  # One row dropped

    def test_load_from_arrays(self, sample_arrays):
        """Test loading from arrays"""
        y, W, Z, group, X = sample_arrays
        dm = DataManager()
        dm.load_from_arrays(
            y=y,
            W=W,
            Z=Z,
            group=group,
            X=X,
            x_cols=['x1', 'x2'],
            z_cols=['z1', 'z2']
        )
        assert dm._data_type_flag == "pandas"
        assert len(dm._data) == len(y)
        assert all(col in dm._data.columns for col in ['x1', 'x2', 'z1', 'z2'])

    def test_load_from_arrays_invalid_shapes(self, sample_arrays):
        """Test loading arrays with inconsistent shapes"""
        y, W, Z, group, X = sample_arrays
        dm = DataManager()

        # Test with mismatched lengths
        with pytest.raises(ValueError):
            dm.load_from_arrays(
                y=y,
                W=W[:-1],  # Wrong length
                Z=Z,
                group=group,
                X=X
            )

    def test_split_data_in_two(self, sample_df):
        """Test splitting data"""
        dm = DataManager()
        dm.load_pandas(
            df=sample_df,
            y_col='target',
            w_col='weight',
            z_cols=['z1', 'z2'],
            group_col='group'
        )

        # Test with default 50-50 split
        df1, df2 = dm.split_data_in_two(random_state=42)
        assert len(df1) + len(df2) == len(sample_df)
        assert abs(len(df1) - len(df2)) <= 1  # Almost equal split

        # Test with custom proportion
        df1, df2 = dm.split_data_in_two(prop=0.8, random_state=42)
        assert len(df1) == int(0.8 * len(sample_df))

    def test_split_data_invalid_prop(self, sample_df):
        """Test splitting data with invalid proportion"""
        dm = DataManager()
        dm.load_pandas(
            df=sample_df,
            y_col='target',
            w_col='weight',
            z_cols=['z1', 'z2'],
            group_col='group'
        )

        with pytest.raises(ValueError):
            dm.split_data_in_two(prop=1.5)

    def test_property_access(self, sample_df):
        """Test property accessor methods"""
        dm = DataManager()

        # Test accessing properties before loading data
        with pytest.raises(ValueError):
            _ = dm.data

        with pytest.raises(ValueError):
            _ = dm.split_indices

        # Test accessing properties after loading data
        dm.load_pandas(
            df=sample_df,
            y_col='target',
            w_col='weight',
            z_cols=['z1', 'z2'],
            group_col='group'
        )

        assert isinstance(dm.data, pd.DataFrame)
        assert dm.data.equals(sample_df)

        # Test columns property
        cols = dm.columns
        assert cols['y'] == 'target'
        assert cols['w'] == 'weight'
        assert cols['z'] == ['z1', 'z2']
        assert cols['group'] == 'group'
