from fastapi import FastAPI,HTTPException
from schema import Info
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR /'model'/ 'sgd_regressor.joblib'


def model_load():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        return 'sorry file not found file not found'
app = FastAPI()
model = model_load()

@app.get('/home')
def home():
    return {'message':'welcome to our page'}


@app.post('/prediction')
def func(info:Info):
    data = info.model_dump()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {'academic performance':prediction.tolist()}

# @app.get('/model_load')
# def model_success():
#     if model:
#         return {'message':'model load successfully'}
#     return None
