import cv2
import torch
import numpy as np
from torchvision import transforms
from facenet_pytorch import MTCNN
import joblib
from torchvision import models
import torch.nn as nn
from PIL import Image


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
# 1. Chargement des modèles pré-entraînés
# -----------------------------
# YOLOv5 pour la détection des visages
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Chargement de YOLOv5

# MTCNN pour affiner la détection des visages
mtcnn = MTCNN(keep_all=False, device=device)

# EfficientNet pour l'extraction des caractéristiques
efficientnet_model = models.efficientnet_b0(pretrained=True)
efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, 7)  # 7 classes
efficientnet_model.load_state_dict(torch.load("best_efficientnetNormal_model.pth"))
efficientnet_model.eval()
efficientnet_model = efficientnet_model.to(device)

# Modèle XGBoost pour la prédiction
best_xgb_model = joblib.load("xgb_efficientnet_modelNormal_best.pkl")

# Scaler pour normalisation
scaler = joblib.load("scaler.pkl")

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
def detect_faces_and_predict(frame):
    """Détecte les visages dans une frame, affine avec MTCNN, et fait des prédictions."""
    results = model_yolo(frame)  # Détection YOLO
    boxes = results.xyxy[0].cpu().numpy()  # Boîtes englobantes

    predictions = []
    for box in boxes:
        x_min, y_min, x_max, y_max, confidence, class_id = map(int, box[:6])
        if class_id == 0:  # ID 0 correspond à 'person'
            cropped_face = frame[y_min:y_max, x_min:x_max]

            # Affiner avec MTCNN
            face_tensor = mtcnn(Image.fromarray(cropped_face))
            if face_tensor is not None:
                # Prétraitement pour le modèle
                face_pil = transforms.ToPILImage()(face_tensor)
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

                predictions.append((prediction[0], prediction_proba, (x_min, y_min, x_max, y_max)))

    return predictions

# -----------------------------
# 4. Capture vidéo en temps réel
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 pour webcam, ou chemin vers une vidéo

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion BGR -> RGB pour les modèles
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection et prédictions
    results = detect_faces_and_predict(frame_rgb)

    # Dessiner les résultats sur l'image
    for predicted_label, probabilities, (x_min, y_min, x_max, y_max) in results:
        class_label = class_names[predicted_label]
        confidence = max(probabilities[0]) * 100

        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Afficher le nom de la classe et la confiance
        text = f"{class_label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Afficher la vidéo en temps réel
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
