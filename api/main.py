from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from datetime import datetime
from google.cloud import firestore
import pandas as pd
import shap
import numpy as np

app = FastAPI(title="CKD Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Cache the sklearn pipeline and SHAP explainer after model load
_pipeline   = None
_explainer  = None
_feat_names = None

def load_explainer():
    global _pipeline, _explainer, _feat_names
    try:
        _pipeline  = mlflow.sklearn.load_model("models:/kidney-disease-baseline@Production")
        preprocessor = _pipeline.named_steps['preprocessor']
        classifier   = _pipeline.named_steps['classifier']
        numeric_features = preprocessor.transformers_[0][2]
        ohe_names = preprocessor.transformers_[1][1].get_feature_names_out(
            preprocessor.transformers_[1][2]
        ).tolist()
        _feat_names = list(numeric_features) + ohe_names
        _explainer  = shap.TreeExplainer(classifier)
        print("SHAP explainer loaded.")
    except Exception as e:
        print(f"Could not load SHAP explainer: {e}")

load_explainer()

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


# Plain-English labels for every feature the model knows about
FEATURE_LABELS = {
    # Numeric
    "age":  "Age",
    "bp":   "Blood Pressure",
    "bgr":  "Blood Glucose",
    "bu":   "Blood Urea",
    "sc":   "Serum Creatinine",
    "sod":  "Sodium Level",
    "pot":  "Potassium Level",
    "hemo": "Haemoglobin",
    "pcv":  "Packed Cell Volume",
    "wc":   "White Blood Cell Count",
    "rc":   "Red Blood Cell Count",
    # Categorical — OHE expands these into feature_value pairs
    "sg_1.005":      "Specific Gravity (very low)",
    "sg_1.01":       "Specific Gravity (low)",
    "sg_1.015":      "Specific Gravity (normal-low)",
    "sg_1.02":       "Specific Gravity (normal)",
    "sg_1.025":      "Specific Gravity (normal-high)",
    "al_0.0":        "Albumin (none)",
    "al_1.0":        "Albumin (trace)",
    "al_2.0":        "Albumin (moderate)",
    "al_3.0":        "Albumin (high)",
    "al_4.0":        "Albumin (very high)",
    "al_5.0":        "Albumin (severe)",
    "su_0.0":        "Sugar (none)",
    "su_1.0":        "Sugar (trace)",
    "su_2.0":        "Sugar (moderate)",
    "su_3.0":        "Sugar (high)",
    "su_4.0":        "Sugar (very high)",
    "su_5.0":        "Sugar (severe)",
    "rbc_abnormal":  "Red Blood Cells (abnormal)",
    "rbc_normal":    "Red Blood Cells (normal)",
    "pc_abnormal":   "Pus Cells (abnormal)",
    "pc_normal":     "Pus Cells (normal)",
    "pcc_present":   "Pus Cell Clumps (present)",
    "pcc_notpresent":"Pus Cell Clumps (absent)",
    "ba_present":    "Bacteria (present)",
    "ba_notpresent": "Bacteria (absent)",
    "htn_yes":       "High Blood Pressure (yes)",
    "htn_no":        "High Blood Pressure (no)",
    "dm_yes":        "Diabetes (yes)",
    "dm_no":         "Diabetes (no)",
    "cad_yes":       "Heart Condition (yes)",
    "cad_no":        "Heart Condition (no)",
    "appet_good":    "Appetite (good)",
    "appet_poor":    "Appetite (poor)",
    "pe_yes":        "Swollen Feet/Ankles (yes)",
    "pe_no":         "Swollen Feet/Ankles (no)",
    "ane_yes":       "Anaemia (yes)",
    "ane_no":        "Anaemia (no)",
}

def humanise(raw_name: str) -> str:
    """Convert a raw model feature name to a plain-English label."""
    return FEATURE_LABELS.get(raw_name, raw_name.replace("_", " ").title())


@app.post("/explain")
def explain(data: PatientData):
    """Returns the top 8 SHAP feature contributions with plain-English labels."""
    if _explainer is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Explainer not loaded")

    input_df = pd.DataFrame([data.model_dump()])
    X_transformed = _pipeline.named_steps['preprocessor'].transform(input_df)

    shap_values = _explainer.shap_values(X_transformed)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    contributions = sv[0]

    pairs = sorted(zip(_feat_names, contributions.tolist()), key=lambda x: abs(x[1]), reverse=True)[:8]
    return {
        "top_features": [
            {"feature": humanise(f), "shap_value": round(v, 4)}
            for f, v in pairs
        ]
    }

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
