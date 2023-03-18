from pydantic import BaseModel
from fastapi import FastAPI
import tensorflow as tf
import pandas as pd
from src.ml.data import process_data
from src.ml.train_model import cat_features
from src.ml.model import inference
import pickle

app = FastAPI()
to_csv = lambda x: x.replace("_", "-")


class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = to_csv

    #     schema_extra = {
    #         "example": {
    #             "name": "Foo",
    #             "description": "A very nice Item",
    #             "price": 35.4,
    #             "tax": 3.2,
    #         }
    #     }


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(input: Input):
    model = tf.keras.models.load_model("src/model/model.h5")
    with open("src/model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("src/model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("src/model/label_binarizer.pkl", "rb") as f:
        label_binarizer = pickle.load(f)

    input_df = pd.DataFrame(input.dict(by_alias=True), index=[0])

    X, y, _, _, _ = process_data(
        input_df, categorical_features=cat_features, training=False, encoder=encoder, scaler=scaler, lb=label_binarizer
    )
    preds = inference(model, X)
    preds = label_binarizer.inverse_transform(preds)
    return {"OK": 200, "prediction": preds[0]}
