from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict():
    # Run prediction using the ML model trained to predict the salary of a person
    # Load ML Model from disk with extension .pb or .h5 using tensorflow or keras
    model = tf.keras.models.load_model(filepath="../model")
    print("model loaded", model.summary())

    return {"OK": 200}
