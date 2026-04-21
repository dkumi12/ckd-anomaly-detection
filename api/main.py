from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from datetime import datetime
from google.cloud import firestore
import pandas as pd

app = FastAPI(title="CKD Prediction API")

mlflow.set_tracking_uri("https://mlflow-tracking-server-330013579477.europe-west1.run.app")

# Initialize Firestore database connection
db = firestore.Client(project="chronic-kd")

# Global variables to hold our model and its metadata in memory
current_model = None
model_metadata = {
    "model_name": "kidney-disease-baseline",
    "version": "unknown",
    "stage": "Production",
    "trained_at": "unknown"
}

def load_production_model():
    """Fetches the Production-aliased model from MLflow Model Registry and loads it into memory."""
    global current_model, model_metadata
    client = MlflowClient()

    try:
        prod_version = client.get_model_version_by_alias("kidney-disease-baseline", "Production")
        current_model = mlflow.pyfunc.load_model("models:/kidney-disease-baseline@Production")
        model_metadata["version"] = prod_version.version
        model_metadata["trained_at"] = prod_version.creation_timestamp
        print(f"Successfully loaded model version {prod_version.version}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load the model when the API starts
load_production_model()

class PatientData(BaseModel):
    # Numeric features
    age:  Optional[float] = None
    bp:   Optional[float] = None
    bgr:  Optional[float] = None
    bu:   Optional[float] = None
    sc:   Optional[float] = None
    sod:  Optional[float] = None
    pot:  Optional[float] = None
    hemo: Optional[float] = None
    pcv:  Optional[float] = None
    wc:   Optional[float] = None
    rc:   Optional[float] = None
    # Categorical features
    sg:    Optional[str] = None
    al:    Optional[str] = None
    su:    Optional[str] = None
    rbc:   Optional[str] = None
    pc:    Optional[str] = None
    pcc:   Optional[str] = None
    ba:    Optional[str] = None
    htn:   Optional[str] = None
    dm:    Optional[str] = None
    cad:   Optional[str] = None
    appet: Optional[str] = None
    pe:    Optional[str] = None
    ane:   Optional[str] = None

@app.post("/predict")
def predict(data: PatientData):
    """Receives input data, runs the model, returns a prediction AND logs it to Firestore"""
    if current_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_dict = data.model_dump()
    input_df = pd.DataFrame([input_dict])

    # Generate prediction
    prediction = current_model.predict(input_df)
    pred_value = prediction.tolist()

    # Log request to Firestore
    log_document = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_version": model_metadata["version"],
        "inputs": input_dict,
        "prediction": pred_value
    }

    try:
        db.collection("prediction_logs").add(log_document)
        print("Successfully logged request to Firestore!")
    except Exception as e:
        print(f"Failed to log to Firestore: {e}")

    return {"prediction": pred_value}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return model_metadata

@app.post("/reload-model")
def reload_model():
    success = load_production_model()
    if success:
        return {"status": "success", "message": f"Reloaded version {model_metadata['version']}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")
