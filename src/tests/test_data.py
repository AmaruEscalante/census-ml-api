import pandas as pd
import numpy as np
from ml.data import process_data
from ml.train_model import cat_features


def test_column_names(data: pd.DataFrame):
    expected_columns = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]

    these_columns = data.columns.values

    assert list(expected_columns) == list(these_columns)


def test_processing_data(data: pd.DataFrame):
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    # Check that the data is the right shape.
    assert X.shape == (32561, 103)
    assert y.shape == (32561,)
    # Check that the data is the right type.
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    # Check that the data has the right values.
    assert np.allclose(X[0, 0], 0.3013698630136986)
