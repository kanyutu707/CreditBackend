from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Any
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

class SafeLabelEncoder:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes_ = None
        self.default_value = None
    
    def fit(self, series):
 
        placeholder = "___UNKNOWN___"

        while placeholder in series.values:
            placeholder += "_"
        
        values = np.append(series.values, placeholder)
        self.encoder.fit(values)
        self.classes_ = self.encoder.classes_
        
        self.default_value = self.encoder.transform([placeholder])[0]
        return self
    
    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)
    
    def transform(self, series):
        result = np.zeros(len(series), dtype=int)
        for i, val in enumerate(series):
            try:
                result[i] = self.encoder.transform([val])[0]
            except ValueError:

                result[i] = self.default_value
        return result
    
    def inverse_transform(self, encodings):
        result = []
        for code in encodings:
            if code == self.default_value:
                result.append("UNKNOWN")
            else:
                try:
                    result.append(self.encoder.inverse_transform([code])[0])
                except ValueError:
                    result.append("UNKNOWN")
        return result

app = FastAPI(title="Fraud Detection API", 
              description="API for detecting fraudulent transactions",
              version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

try:
    with open("model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None


label_encoders = {}
try:
    with open("label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    print("Label encoders loaded successfully!")
    

    for col, encoder in label_encoders.items():
        if not isinstance(encoder, SafeLabelEncoder):
            safe_encoder = SafeLabelEncoder()
            safe_encoder.encoder.classes_ = encoder.classes_
            safe_encoder.classes_ = encoder.classes_
            safe_encoder.default_value = len(encoder.classes_)
            label_encoders[col] = safe_encoder
except Exception as e:
    print(f"Label encoders not found, will create new ones: {e}")

class TransactionRequest(BaseModel):
    trans_date_trans_time: str
    cc_num: float
    merchant: str
    category: str
    amt: float
    first: str
    last: str
    gender: str
    street: str
    city: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    job: str
    dob: str
    trans_num: str
    unix_time: int
    merch_lat: float
    merch_long: float
    
    class Config:
        schema_extra = {
            "example": {
                "trans_date_trans_time": "2019-01-01 00:00:00",
                "cc_num": 3.560730e+15,
                "merchant": "fraud_Rippin, Kub and Mann",
                "category": "misc_net",
                "amt": 24.84,
                "first": "Jennifer",
                "last": "Banks",
                "gender": "F",
                "street": "561 Perry Cove",
                "city": "Moravian Falls",
                "state": "NC",
                "zip": 28654,
                "lat": 31.8599,
                "long": -102.7413,
                "city_pop": 3495,
                "job": "Psychologist, counselling",
                "dob": "1988-03-09",
                "trans_num": "2e12ohf49dsb2e12ohf49dsc",
                "unix_time": 1371852399,
                "merch_lat": 32.575873,
                "merch_long": -102.60429
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    is_fraud: bool
    confidence: Optional[float] = None
    unknown_values: Dict[str, Any] = {}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fraud Detection API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    if loaded_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
       
        input_data = pd.DataFrame([transaction.dict()])
        
        input_data.insert(0, "Unnamed: 0", [0])
        

        unknown_values = {}
        
     
        categorical_columns = [
            'trans_date_trans_time', 'merchant', 'category', 
            'first', 'last', 'gender', 'street', 'city', 
            'state', 'job', 'dob', 'trans_num'
        ]
        
      
        for col in categorical_columns:
            original_value = input_data[col].iloc[0]
            
            if col in label_encoders:

                if not isinstance(label_encoders[col], SafeLabelEncoder):
                    
                    encoder = SafeLabelEncoder()
                    encoder.encoder.classes_ = label_encoders[col].classes_
                    encoder.classes_ = label_encoders[col].classes_
                    encoder.default_value = len(label_encoders[col].classes_)
                    label_encoders[col] = encoder
                
          
                if original_value not in label_encoders[col].classes_:
                    unknown_values[col] = original_value
                
               
                input_data[col] = label_encoders[col].transform(input_data[col])
            else:
               
                encoder = SafeLabelEncoder()
                input_data[col] = encoder.fit_transform(input_data[col])
                label_encoders[col] = encoder
   
        prediction = loaded_model.predict(input_data)

        confidence = None
        if hasattr(loaded_model, "predict_proba"):
            try:
                proba = loaded_model.predict_proba(input_data)
                confidence = float(proba[0][1])  
            except:
                pass
                
        return {
            "prediction": int(prediction[0]),
            "is_fraud": bool(prediction[0]),
            "confidence": confidence,
            "unknown_values": unknown_values
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9030, reload=True)