from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import json
import pickle
from pathlib import Path

import pandas as pd
import gradio as gr


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py:
# mlops-aws-learning/
#     src/
#         app/
#             app.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "deployment" / "model"

MODEL_FILE = MODEL_DIR / "model.pkl"
FEATURE_FILE = MODEL_DIR / "feature_columns.json"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

SELECTED_MODEL_FILE = PROJECT_ROOT / "selected_model.json"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Telco Customer Churn API",
    description="Telco churn prediction using the selected MLflow model",
    version="1.0.0"
)


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

print("\n" + "=" * 60)
print("🚀 LOADING SELECTED MODEL")
print("=" * 60)

print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"MODEL_DIR    : {MODEL_DIR}")


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    MODEL_FILE,
    FEATURE_FILE,
    METADATA_FILE
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"\n❌ Required file not found:\n{file}\n\n"
            f"Run:\n"
            f"python src\\serving\\deployment\\download_model.py"
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("\n📦 Loading model.pkl...")

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully")


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

print("\n📋 Loading feature_columns.json...")

with open(FEATURE_FILE, "r") as f:
    feature_columns = json.load(f)

print(
    f"✅ Loaded {len(feature_columns)} feature columns"
)

print("\nModel features:")
for feature in feature_columns:
    print(f"  - {feature}")


# ============================================================
# LOAD MODEL METADATA
# ============================================================

print("\n📄 Loading model_metadata.json...")

with open(METADATA_FILE, "r") as f:
    model_metadata = json.load(f)

print("✅ Model metadata loaded")


# ============================================================
# LOAD SELECTED MODEL INFORMATION
# ============================================================

selected_model = {}

if SELECTED_MODEL_FILE.exists():

    with open(SELECTED_MODEL_FILE, "r") as f:
        selected_model = json.load(f)

    print("\n🏆 SELECTED MODEL")
    print("-" * 60)

    print(
        f"Model   : "
        f"{selected_model.get('model_name', 'N/A')}"
    )

    print(
        f"Version : "
        f"{selected_model.get('version', 'N/A')}"
    )

    print(
        f"Run ID  : "
        f"{selected_model.get('run_id', 'N/A')}"
    )

    print(
        f"ROC-AUC : "
        f"{selected_model.get('roc_auc', 'N/A')}"
    )


print("\n" + "=" * 60)
print("✅ MODEL READY FOR INFERENCE")
print("=" * 60)


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_input(payload: dict) -> pd.DataFrame:

    """
    Convert raw customer input into the exact feature format
    expected by the trained model.
    """

    print("\n" + "-" * 60)
    print("🔄 PREPROCESSING INPUT")
    print("-" * 60)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([payload])

    print("\nOriginal columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # One-hot encode categorical columns
    # --------------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True
    )

    # --------------------------------------------------------
    # Convert boolean columns to numeric
    # --------------------------------------------------------

    for col in df.columns:

        if df[col].dtype == bool:

            df[col] = df[col].astype(int)

    # --------------------------------------------------------
    # Make sure all model features exist
    # --------------------------------------------------------

    for col in feature_columns:

        if col not in df.columns:

            df[col] = 0

    # --------------------------------------------------------
    # Remove unexpected columns
    # --------------------------------------------------------

    df = df[feature_columns]

    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    df = df.astype(float)

    print("\nFinal model columns:")
    print(df.columns.tolist())

    print("\nFinal input shape:")
    print(df.shape)

    return df


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Telco Customer Churn API",
        "model_loaded": True
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_name": selected_model.get(
            "model_name",
            "telco-churn-model"
        ),
        "version": selected_model.get(
            "version",
            "unknown"
        ),
        "run_id": selected_model.get(
            "run_id",
            "unknown"
        ),
        "roc_auc": selected_model.get(
            "roc_auc",
            "unknown"
        ),
        "feature_count": len(feature_columns),
        "model_directory": str(MODEL_DIR)
    }


# ============================================================
# REQUEST SCHEMA
# ============================================================

class CustomerData(BaseModel):

    # --------------------------------------------------------
    # Customer information
    # --------------------------------------------------------

    gender: str
    SeniorCitizen: int

    Partner: str
    Dependents: str

    # --------------------------------------------------------
    # Phone
    # --------------------------------------------------------

    PhoneService: str
    MultipleLines: str

    # --------------------------------------------------------
    # Internet
    # --------------------------------------------------------

    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    # --------------------------------------------------------
    # Streaming
    # --------------------------------------------------------

    StreamingTV: str
    StreamingMovies: str

    # --------------------------------------------------------
    # Contract / billing
    # --------------------------------------------------------

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    tenure: int
    MonthlyCharges: float
    TotalCharges: float


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def api_predict(data: CustomerData):

    try:

        # ----------------------------------------------------
        # Convert request to dictionary
        # ----------------------------------------------------

        payload = data.model_dump()

        print("\n" + "=" * 60)
        print("🔮 NEW PREDICTION REQUEST")
        print("=" * 60)

        print("\nReceived payload:")
        print(payload)

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        df = preprocess_input(payload)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(df)[0]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        churn_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(df)

            churn_probability = float(
                probabilities[0][1]
            )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {
            "status": "SUCCESS",
            "prediction": int(prediction),
            "churn": bool(prediction),
            "churn_probability": churn_probability
        }

        print("\n" + "=" * 60)
        print("✅ PREDICTION")
        print("=" * 60)

        print(result)

        return result

    except Exception as e:

        print("\n❌ Prediction error:")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# GRADIO INTERFACE FUNCTION
# ============================================================

def gradio_interface(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    tenure,
    MonthlyCharges,
    TotalCharges
):

    payload = {

        "gender": gender,

        "SeniorCitizen": int(SeniorCitizen),

        "Partner": Partner,

        "Dependents": Dependents,

        "PhoneService": PhoneService,

        "MultipleLines": MultipleLines,

        "InternetService": InternetService,

        "OnlineSecurity": OnlineSecurity,

        "OnlineBackup": OnlineBackup,

        "DeviceProtection": DeviceProtection,

        "TechSupport": TechSupport,

        "StreamingTV": StreamingTV,

        "StreamingMovies": StreamingMovies,

        "Contract": Contract,

        "PaperlessBilling": PaperlessBilling,

        "PaymentMethod": PaymentMethod,

        "tenure": int(tenure),

        "MonthlyCharges": float(MonthlyCharges),

        "TotalCharges": float(TotalCharges),
    }

    # --------------------------------------------------------
    # SAME preprocessing as API
    # --------------------------------------------------------

    df = preprocess_input(payload)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(df)[0]

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(df)

        probability = float(
            probabilities[0][1]
        )

    return {
        "prediction": int(prediction),
        "churn": bool(prediction),
        "churn_probability": probability
    }


# ============================================================
# GRADIO INTERFACE
# ============================================================

demo = gr.Interface(

    fn=gradio_interface,

    inputs=[

        # ----------------------------------------------------
        # Customer
        # ----------------------------------------------------

        gr.Dropdown(
            ["Male", "Female"],
            label="Gender"
        ),

        gr.Number(
            value=0,
            label="Senior Citizen"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Partner"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Dependents"
        ),

        # ----------------------------------------------------
        # Phone
        # ----------------------------------------------------

        gr.Dropdown(
            ["Yes", "No"],
            label="Phone Service"
        ),

        gr.Dropdown(
            ["Yes", "No", "No phone service"],
            label="Multiple Lines"
        ),

        # ----------------------------------------------------
        # Internet
        # ----------------------------------------------------

        gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            label="Internet Service"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Online Security"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Online Backup"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Device Protection"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Tech Support"
        ),

        # ----------------------------------------------------
        # Streaming
        # ----------------------------------------------------

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming TV"
        ),

        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming Movies"
        ),

        # ----------------------------------------------------
        # Contract / billing
        # ----------------------------------------------------

        gr.Dropdown(
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            label="Contract"
        ),

        gr.Dropdown(
            ["Yes", "No"],
            label="Paperless Billing"
        ),

        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            label="Payment Method"
        ),

        # ----------------------------------------------------
        # Numerical
        # ----------------------------------------------------

        gr.Number(
            label="Tenure (months)"
        ),

        gr.Number(
            label="Monthly Charges"
        ),

        gr.Number(
            label="Total Charges"
        ),
    ],

    outputs="json",

    title="Telco Churn Predictor",

    description=(
        "Enter customer details to predict "
        "whether the customer is likely to churn."
    )
)


# ============================================================
# MOUNT GRADIO
# ============================================================

app = gr.mount_gradio_app(
    app,
    demo,
    path="/ui"
)


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )