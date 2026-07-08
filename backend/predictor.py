"""
Prediction Engine
KrishiVaidya
"""

import torch
import torch.nn as nn
from torchvision import models

from backend.config import MODEL_PATH, DEVICE


class PlantDiseasePredictor:
    """
    Loads the trained model and performs prediction.
    """

    def __init__(self):

        # --------------------------
        # Load checkpoint
        # --------------------------
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        self.class_names = checkpoint["class_names"]
        self.num_classes = checkpoint["num_classes"]
        self.image_size = checkpoint["image_size"]

        print("✅ Checkpoint Loaded")

        # --------------------------
        # Build EfficientNet-B0
        # --------------------------
        self.model = models.efficientnet_b0(weights=None)

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            self.num_classes
        )

        # --------------------------
        # Load trained weights
        # --------------------------
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)

        self.model.eval()

        print("✅ Model Loaded Successfully")


if __name__ == "__main__":

    predictor = PlantDiseasePredictor()

    print()

    print("Classes :", predictor.num_classes)

    print("Image Size :", predictor.image_size)

    print()

    print("First 5 Classes")

    for cls in predictor.class_names[:5]:
        print("-", cls)