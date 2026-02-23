from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


model=joblib.load('house_prices.joblib')
feature_name= joblib.load('feature_names.joblib')

house=FastAPI(title="House Prices API")

class HouseFeatures(BaseModel):
    size_sqft:float
    bedrooms:int
    bathrooms:int
    
@house.get('/')
def root():
    return{"message":"house prices api online!!!!"}  

@house.post('/predict')
def predict_sales(features:HouseFeatures):
    x=np.array([[getattr(features,f) for f in feature_name]])
    prediction =model.predict(x)[0]
    return{"predicted_sales":float(prediction)}
    
    