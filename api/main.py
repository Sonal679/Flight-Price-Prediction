from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Flight Price Prediction API")

# Load trained pipeline model
model = joblib.load("model/flight_price_model.pkl")


class FlightData(BaseModel):
    Airline: str
    Source: str
    Destination: str

    Total_Stops: int
    Journey_Day: int
    Journey_Month: int

    Dep_Hour: int
    Dep_Min: int

    Arrival_Hour: int
    Arrival_Min: int

    Duration_Hours: int
    Duration_Mins: int


@app.get("/")
def home():
    return {"message": "Flight Price Prediction API is running successfully!"}


@app.post("/predict")
def predict(data: FlightData):

    input_data = pd.DataFrame([{
        "Airline": data.Airline,
        "Source": data.Source,
        "Destination": data.Destination,
        "Total_Stops": data.Total_Stops,
        "Journey_Day": data.Journey_Day,
        "Journey_Month": data.Journey_Month,
        "Dep_Hour": data.Dep_Hour,
        "Dep_Min": data.Dep_Min,
        "Arrival_Hour": data.Arrival_Hour,
        "Arrival_Min": data.Arrival_Min,
        "Duration_Hours": data.Duration_Hours,
        "Duration_Mins": data.Duration_Mins
    }])

    prediction = model.predict(input_data)

    return {
        "Predicted Flight Price": round(float(prediction[0]), 2)
    }