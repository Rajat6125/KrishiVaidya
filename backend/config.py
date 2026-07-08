"""
Project Configuration
KrishiVaidya
"""

from pathlib import Path
import torch

# ==========================
# Project Paths
# ==========================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "best_model.pth"

ASSETS_DIR = ROOT_DIR / "assets"

DATA_DIR = ROOT_DIR / "data"

# ==========================
# Device Configuration
# ==========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)