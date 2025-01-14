from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from facenet_pytorch import MTCNN
import joblib
from PIL import Image
import torch.nn as nn

# Initialisation
app = Flask(__name__)

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

# Détection sur CPU
device = torch.device("cpu")  # Force l'utilisation du CPU

# Chargement des modèles
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)  # YOLOv5
mtcnn = MTCNN(keep_all=False, device=device)  # MTCNN sur CPU
efficientnet_model = models.efficientnet_b0(pretrained=True)
efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, 7)  # 7 classes
efficientnet_model.load_state_dict(torch.load("best_efficientnetNormal_model.pth", map_location=device))
efficientnet_model.eval().to(device)  # Modèle en mode évaluation sur CPU
best_xgb_model = joblib.load("xgb_efficientnet_modelNormal_best.pkl")
scaler = joblib.load("scaler.pkl")

# Transformation pour EfficientNet
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_faces_and_predict(frame):
    """Détecte les visages dans une image et prédit les émotions."""
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

                # Extraction des caractéristiques
                with torch.no_grad():
                    features = efficientnet_model.features(face_transformed)
                    features = efficientnet_model.avgpool(features)
                    features = torch.flatten(features, 1).cpu().numpy()

                # Normalisation et prédiction
                features_scaled = scaler.transform(features)
                prediction = best_xgb_model.predict(features_scaled)
                prediction_proba = best_xgb_model.predict_proba(features_scaled)

                predictions.append({
                    "label": class_names[prediction[0]],
                    "confidence": float(max(prediction_proba[0])),
                    "box": [x_min, y_min, x_max, y_max]
                })

    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour la détection d'émotions."""
    if 'image' not in request.files:
        return jsonify({"error": "Veuillez envoyer une image."}), 400

    # Charger l'image envoyée
    file = request.files['image']
    image_path = "./temp_image.jpg"
    file.save(image_path)

    # Lecture et conversion de l'image
    frame = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection et prédictions
    predictions = detect_faces_and_predict(frame_rgb)

    return jsonify({"predictions": predictions})

# Lancer le serveur
if __name__ == '__main__':
    app.run(debug=True)
