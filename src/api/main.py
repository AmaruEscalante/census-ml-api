from pydantic import BaseModel
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()

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

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(input: Input):
    model = tf.keras.models.load_model(filepath="../model")
    
    print(input)

    return {"OK": 200}