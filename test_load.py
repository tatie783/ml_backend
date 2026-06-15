import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = "psmear_model.pth"
CLASS_NAMES = ["Leukemia", "Lymphoma", "Normal"]

device = torch.device("cpu")
model = models.mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("SUCCESS: Model loaded successfully.")
except Exception as e:
    print(f"FAILURE: {e}")
