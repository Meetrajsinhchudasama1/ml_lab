from fastapi import FastAPI
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load("artifacts/pipeline.pkl")

app = FastAPI(title="ML Model API", description="Deploy trained ML pipeline")

@app.get("/")
def home():
    return {"message": "ML model API is running!"}

@app.post("/predict")
def predict(age: float, salary: float, department: str, tenure: float):
    # Create input dataframe
    data = pd.DataFrame([{
        "age": age,
        "salary": salary,
        "department": department,
        "tenure": tenure
    }])
    # Run prediction
    prediction = model.predict(data)[0]
    return {"prediction": int(prediction)}