import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
from facenet_pytorch import MTCNN
import cv2

# Dictionnaire des noms des classes
class_names = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral"
}

# Initialisation de l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# -----------------------------
# 1. Chargement du modèle YOLOv5 via PyTorch Hub
# -----------------------------
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Chargement du modèle YOLOv5 pré-entraîné

# -----------------------------
# 2. Transformation utilisée pour le modèle
# -----------------------------
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -----------------------------
# 3. Détection des visages avec YOLO et MTCNN
# -----------------------------
mtcnn = MTCNN(keep_all=False, device=device)

def detect_faces_with_yolo_and_mtcnn(image_path):
    """Détecte les visages avec YOLO (PyTorch Hub), puis affine avec MTCNN."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Détection des visages avec YOLO
    results = model(img_rgb)
    boxes = results.xyxy[0].cpu().numpy()

    faces = []
    for box in boxes:
        x_min, y_min, x_max, y_max, confidence, class_id = map(int, box[:6])
        if class_id == 0:  # ID 0 correspond à 'person'
            cropped_face = img_rgb[y_min:y_max, x_min:x_max]

            # Affiner avec MTCNN
            face_tensor = mtcnn(Image.fromarray(cropped_face))
            if face_tensor is not None:
                faces.append(face_tensor)

    return faces  # Retourne les visages sous forme de Tensor

# -----------------------------
# 4. Prédiction avec les modèles pré-entraînés
# -----------------------------
def preprocess_and_predict(image_path):
    """Prédiction sur une image donnée après détection et prétraitement."""
    faces = detect_faces_with_yolo_and_mtcnn(image_path)
    if not faces:
        print(f"Aucun visage détecté dans {image_path}")
        return None

    predictions = []
    for face in faces:
        # Convertir le Tensor en image PIL
        face_pil = transforms.ToPILImage()(face)

        # Prétraitement pour le modèle
        face_transformed = transform_test(face_pil).unsqueeze(0).to(device)

        # Extraction des caractéristiques avec EfficientNet
        with torch.no_grad():
            features = efficientnet_model.features(face_transformed)
            features = efficientnet_model.avgpool(features)
            features = torch.flatten(features, 1).cpu().numpy()

        # Normalisation et prédiction
        features_scaled = scaler.transform(features)
        prediction = best_xgb_model.predict(features_scaled)
        prediction_proba = best_xgb_model.predict_proba(features_scaled)

        predictions.append((prediction[0], prediction_proba))

    return predictions

# -----------------------------
# 5. Chargement des modèles EfficientNet et XGBoost
# -----------------------------
print("Chargement des modèles pré-entraînés...")
efficientnet_model = models.efficientnet_b0(pretrained=True)
efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, 7)  # Nombre de classes

efficientnet_model.load_state_dict(torch.load("best_efficientnetNormal_model.pth"))
efficientnet_model.eval()
efficientnet_model = efficientnet_model.to(device)

best_xgb_model = joblib.load("xgb_efficientnet_modelNormal_best.pkl")
scaler = joblib.load("scaler.pkl")  # Assurez-vous d'avoir sauvegardé le scaler utilisé pendant l'entraînement

# -----------------------------
# 6. Tester avec des nouvelles images
# -----------------------------
new_image_path = r"C:\Users\siham\OneDrive\Bureau\ECL\MLyon 1\Projet\ProjectML\surpris.jpg"  # Remplacez par le chemin de l'image à tester
result = preprocess_and_predict(new_image_path)

if result:
    for idx, (predicted_label, probabilities) in enumerate(result):
        class_label = class_names[predicted_label]  # Convertir le numéro de classe en label textuel
        print(f"Visage {idx + 1}:")
        print(f"  Prédiction : {class_label}")
        print(f"  Probabilités : {probabilities}")
