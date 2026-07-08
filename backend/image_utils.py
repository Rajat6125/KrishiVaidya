"""
Image Processing Utilities
KrishiVaidya

This module is responsible for:
1. Loading an image
2. Converting it to RGB
3. Applying preprocessing transforms
4. Returning a PyTorch tensor
"""

from PIL import Image
from torchvision import transforms

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageProcessor:
    """
    Handles all image preprocessing for prediction.
    """

    def __init__(self, image_size: int):

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        ])

    def preprocess(self, image_path: str):
        """
        Loads an image and converts it into
        a model-ready tensor.

        Parameters
        ----------
        image_path : str

        Returns
        -------
        torch.Tensor
        """

        image = Image.open(image_path).convert("RGB")

        tensor = self.transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor