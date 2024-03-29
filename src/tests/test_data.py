import pandas as pd
import numpy as np
from src.ml.data import process_data
from src.ml.train_model import cat_features
from src.ml.model import train_model, compute_model_metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score


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
    X, y, _, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    # Check that the data is the right shape.
    assert X.shape == (32561, 108)
    assert y.shape == (32561,)
    # Check that the data is the right type.
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_similar_data_distribution(data: pd.DataFrame):
    dist1 = data["education"].value_counts().sort_index()
    # Get the variance, mean and standard deviation of the distribution.
    mean1 = dist1.mean()  # 2035.0625
    std1 = dist1.std()  # 3006.9181669110985
    # Assert that the mean is within a range
    assert 1000 < mean1 < 3000
    # Assert that the standard deviation is within a range
    assert 2500 < std1 < 3500


def test_train_model():
    # Generate some random data
    X, y = make_classification(n_samples=1000, n_features=108, n_informative=5, random_state=42)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)

    assert y_pred.shape == (200, 1), f"Expected y_pred shape ({y_test.shape}), but got {y_pred.shape}"
    assert precision_score(y_test, y_pred) > 0.5, "Precision score should be > 0.5"
    assert recall_score(y_test, y_pred) > 0.5, "Recall score should be > 0.5"
    assert fbeta_score(y_test, y_pred, beta=1) > 0.5, "F1 score should be > 0.5"


def test_compute_model_metrics():
    # Test case 1: Perfect predictions
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    expected_precision = 1.0
    expected_recall = 1.0
    expected_fbeta = 1.0
    actual_precision, actual_recall, actual_fbeta = compute_model_metrics(y_true, y_pred)
    assert np.isclose(
        actual_precision, expected_precision
    ), f"Expected precision {expected_precision}, but got {actual_precision}"
    assert np.isclose(actual_recall, expected_recall), f"Expected recall {expected_recall}, but got {actual_recall}"
    assert np.isclose(actual_fbeta, expected_fbeta), f"Expected F1 {expected_fbeta}, but got {actual_fbeta}"

    # Test case 2: Random predictions
    y_true = np.random.randint(2, size=100)
    y_pred = np.random.randint(2, size=100)
    expected_precision = precision_score(y_true, y_pred, zero_division=1)
    expected_recall = recall_score(y_true, y_pred, zero_division=1)
    expected_fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
    actual_precision, actual_recall, actual_fbeta = compute_model_metrics(y_true, y_pred)
    assert np.isclose(
        actual_precision, expected_precision
    ), f"Expected precision {expected_precision}, but got {actual_precision}"
    assert np.isclose(actual_recall, expected_recall), f"Expected recall {expected_recall}, but got {actual_recall}"
    assert np.isclose(actual_fbeta, expected_fbeta), f"Expected F1 {expected_fbeta}, but got {actual_fbeta}"
