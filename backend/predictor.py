"""
Prediction Engine
KrishiVaidya
"""

import torch
import torch.nn as nn
from torchvision import models

from backend.config import MODEL_PATH, DEVICE
from backend.image_utils import ImageProcessor


class PlantDiseasePredictor:

    def __init__(self):

        # -------------------------
        # Load Checkpoint
        # -------------------------

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        self.class_names = checkpoint["class_names"]
        self.num_classes = checkpoint["num_classes"]
        self.image_size = checkpoint["image_size"]

        # -------------------------
        # Image Processor
        # -------------------------

        self.processor = ImageProcessor(self.image_size)

        # -------------------------
        # Build Model
        # -------------------------

        self.model = models.efficientnet_b0(weights=None)

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            self.num_classes
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)

        self.model.eval()

        print("✅ Model Ready")

    # ==================================================

    def predict(self, image_path):

        image = self.processor.preprocess(image_path)

        image = image.to(DEVICE)

        with torch.no_grad():

            outputs = self.model(image)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probabilities, dim=1)

        prediction = self.class_names[predicted.item()]

        return {
            "prediction": prediction,
            "confidence": round(confidence.item() * 100, 2)
        }


if __name__ == "__main__":

    predictor = PlantDiseasePredictor()

    image_path = "data/sample_images/test.jpg"

    result = predictor.predict(image_path)

    print(result)