import torch
import torch.nn as nn
from torchvision import models
import os

# CONFIG
PTH_MODEL = "psmear_model.pth"
ONNX_MODEL = "psmear_model.onnx"
NUM_CLASSES = 4 # Leukemia, Lymphoma, Myeloma, Normal

def convert():
    if not os.path.exists(PTH_MODEL):
        print(f"Error: {PTH_MODEL} not found. Please run this in the same folder as your model.")
        return

    print("Loading PyTorch model...")
    device = torch.device("cpu")
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(PTH_MODEL, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    print(f"Converting to {ONNX_MODEL}...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, ONNX_MODEL, 
                      input_names=['input'], 
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    print(f"Success! Final size: {os.path.getsize(ONNX_MODEL)/1024/1024:.2f} MB")
    print("You can now upload this .onnx file to PythonAnywhere.")

if __name__ == "__main__":
    convert()
