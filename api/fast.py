from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from TaxiFareModel.params import BUCKET_NAME, MODEL_NAME, MODEL_VERSION
import pandas as pd
import joblib
import os
from google.cloud import storage
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

PATH_TO_LOCAL_MODEL = 'model.joblib'

# Definition to retrieve model from Google Cloud storage
def download_model(model_directory=MODEL_VERSION, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}/versions/{}/{}'.format(
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print("=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model

@app.get("/predict_fare")
def index(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_latitude, dropoff_longitude, passenger_count ):
    params = {
        'key': ['2015-01-27 13:08:24.0000002'],
        'pickup_datetime': [pickup_datetime],
        'pickup_longitude': [float(pickup_longitude)],
        'pickup_latitude': [float(pickup_latitude)],
         'dropoff_longitude': [float(dropoff_longitude)],
        'dropoff_latitude' : [float(dropoff_latitude)],
        'passenger_count': [int(passenger_count)],
    }

    X_pred = pd.DataFrame.from_dict(params)
    print(X_pred.shape)
    # model = download_model(model_directory=MODEL_VERSION, bucket=BUCKET_NAME, rm=True)
    model = joblib.load('model.joblib')
    y_pred = model.predict(X_pred)
    print(y_pred)
    y_pred = y_pred.tolist()[0]

    return {
            "pickup_datetime":pickup_datetime,
            "pickup_longitude":pickup_longitude,
            "pickup_latitude":pickup_latitude,
            "dropoff_longitude":dropoff_longitude,
            "dropoff_latitude":dropoff_latitude,
            "passenger_count":passenger_count,
            "prediction":y_pred
            }
