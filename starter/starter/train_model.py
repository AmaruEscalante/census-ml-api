# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
import numpy as np

# Add code to load in the data.
data = pd.read_csv("data/census.csv")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
processed_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(processed_data, train_size=0.80, test_size=0.20, random_state=42)
X_train, y_train = train[:, :-1], train[:, -1]
# Train and save a model.
model = train_model(X_train, y_train)

X_test, y_test = test[:, :-1], test[:, -1]
preds = inference(model, X_test)
# Compute the model's metrics and print them out.
precision, recall, fbeta = compute_model_metrics(y_test, preds)

