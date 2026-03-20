from fastapi import FastAPI
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# ----------------------------------
# MLflow tracking path
# ----------------------------------

mlflow.set_tracking_uri("file:///C:/Users/Sparsh Sopory/energy_project/mlruns")

client = MlflowClient()

runs = client.search_runs(
    experiment_ids=["878416038388553335"],
    order_by=["metrics.MAE ASC"]
)

if len(runs) == 0:
    raise Exception("No MLflow runs found.")

best_run = runs[0]
best_run_id = best_run.info.run_id

print("Best run:", best_run_id)

model_uri = f"runs:/{best_run_id}/model"

model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully")

# ----------------------------------
# FastAPI
# ----------------------------------

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Energy Sustainability Prediction API running"}

@app.post("/predict")
def predict(sample: dict):
    df = pd.DataFrame([sample])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}