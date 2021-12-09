import pyrebase
from fastapi import FastAPI
from ML_functions import *
from decouple import  config

app = FastAPI()

config = {
    "apiKey": config("apiKey"),
    "authDomain": config("authDomain"),
    "databaseURL": config("databaseURL"),
    "storageBucket": config("storageBucket"),
}

firebase = pyrebase.initialize_app(config)


@app.get("/")
def root():
    return {"Music App": "Welcome to Music App"}


@app.get("/user/get_mood")
def get_mood():
    storage = firebase.storage()
    storage.child("tracks/recordingAudio.mp3").download("ml_model/data.mp3")
    audio_link = "ml_model/data.mp3"
    data = prepare_data(audio_link, n=n_mfcc)
    mean, std = load_mean_std()
    data = (data - mean) / std
    model = tf.keras.models.load_model('ml_model/my_model')
    probs = model.predict(np.expand_dims(data, axis=0))
    return {
        "status": "success",
        "class ": class_mapping[np.argmax(probs)]
    }
