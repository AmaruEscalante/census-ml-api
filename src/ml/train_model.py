# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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


def main():
    from data import process_data
    from model import train_model, compute_model_metrics, inference, compute_slice_performance

    # Add code to load in the data.
    data = pd.read_csv("src/data/census.csv")
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    processed_data = np.concatenate((X, y.reshape(-1, 1)), axis=1)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(processed_data, train_size=0.80, test_size=0.20, random_state=42)
    X_train, y_train = train[:, :-1], train[:, -1]
    # Train and save a model.
    model = train_model(X_train, y_train)

    # Save model.
    model.save("src/model")

    X_test, y_test = test[:, :-1], test[:, -1]
    preds = inference(model, X_test)
    # Compute the model's metrics and print them out.
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    # Compute model metrics on slices of the data.
    data = data.sort_values(by="education")
    X, y, enconder, label_binarizer = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )

    # Binarize salary
    data["salary"] = label_binarizer.transform(data["salary"])
    preds = inference(model, X)

    # Append predictions to data as a new column.
    data["preds"] = preds

    # Compute slice performance.
    compute_slice_performance(data, "education")

    # Print metrics.
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Fbeta b=1: {fbeta}")


if __name__ == "__main__":
    main()
