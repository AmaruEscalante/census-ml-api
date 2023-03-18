from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np
import pandas as pd


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    inputs = tf.keras.Input(shape=(108,))
    x = tf.keras.layers.Dense(16, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2, seed=42)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2, seed=42)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    metrics = [tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")]

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

    batch_size = 32
    epochs = 26

    model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, verbose=True)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_slice_performance(df: pd.DataFrame, feature: str):
    unique_values = df[feature].unique()
    with open("slice_output.txt", "w") as f:
        for value in unique_values:
            slice_df = df[df[feature] == value]
            y_true = slice_df["salary"]
            y_pred = slice_df["preds"]
            cm = confusion_matrix(y_true, y_pred)
            precision = np.diag(cm) / np.sum(cm, axis=0)
            recall = np.diag(cm) / np.sum(cm, axis=1)
            f1_score = 2 * precision * recall / (precision + recall)
            accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            f.write(f"Metrics for {feature}={value}:\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1_score}\n")
            f.write(f"Accuracy: {accuracy}\n\n")


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    preds = model.predict(X)
    return preds
