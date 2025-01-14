from fastapi import FastAPI, UploadFile
import torch
from PIL import Image
import numpy as np
import joblib

app = FastAPI()

# Charger les modèles
efficientnet_model = torch.load(".\\best_efficientnetNormal_model.pth", map_location="cpu")
scaler = joblib.load(".\\scaler.pkl")
xgb_model = joblib.load(".\\xgb_efficientnet_modelNormal_best.pkl")

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    # Ajouter traitement et prédictions
    return {"result": "Prediction here"}
