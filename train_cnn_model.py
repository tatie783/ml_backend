import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Define Paths
BASE_DIR = r"C:\Users\Andriana  Musonza\.gemini\antigravity\scratch"
LEUKEMIA_ALL_DIR = os.path.join(BASE_DIR, r"Leukemia_images\training_data\fold_0\all")
LEUKEMIA_HEM_DIR = os.path.join(BASE_DIR, r"Leukemia_images\training_data\fold_0\hem")
LYMPHOMA_CLL_DIR = os.path.join(BASE_DIR, r"lymphoma_images\CLL")
LYMPHOMA_FL_DIR = os.path.join(BASE_DIR, r"lymphoma_images\FL")
LYMPHOMA_MCL_DIR = os.path.join(BASE_DIR, r"lymphoma_images\MCL")
MYELOMA_DIR = os.path.join(os.path.dirname(__file__), r"myeloma_images\PCMMD Plasma Cells for Multiple Myeloma Diagnosis\data\detection\train\images")

CONSOLIDATED_DATA_DIR = os.path.join(BASE_DIR, "consolidated_dataset")

def prepare_dataset():
    print("Preparing dataset...")
    for cls in ["Leukemia", "Lymphoma", "Normal", "Myeloma"]:
        os.makedirs(os.path.join(CONSOLIDATED_DATA_DIR, cls), exist_ok=True)
        
    if os.path.exists(LEUKEMIA_ALL_DIR):
        for f in os.listdir(LEUKEMIA_ALL_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                shutil.copy2(os.path.join(LEUKEMIA_ALL_DIR, f), os.path.join(CONSOLIDATED_DATA_DIR, "Leukemia", f))
                
    for src_dir in [LYMPHOMA_CLL_DIR, LYMPHOMA_FL_DIR, LYMPHOMA_MCL_DIR]:
        if os.path.exists(src_dir):
            for f in os.listdir(src_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    new_name = os.path.basename(src_dir) + "_" + f
                    shutil.copy2(os.path.join(src_dir, f), os.path.join(CONSOLIDATED_DATA_DIR, "Lymphoma", new_name))
                    
    if os.path.exists(LEUKEMIA_HEM_DIR):
        for f in os.listdir(LEUKEMIA_HEM_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                shutil.copy2(os.path.join(LEUKEMIA_HEM_DIR, f), os.path.join(CONSOLIDATED_DATA_DIR, "Normal", f))
                
    if os.path.exists(MYELOMA_DIR):
        for f in os.listdir(MYELOMA_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                shutil.copy2(os.path.join(MYELOMA_DIR, f), os.path.join(CONSOLIDATED_DATA_DIR, "Myeloma", f))
                
    print("Dataset prepared in:", CONSOLIDATED_DATA_DIR)

def build_and_train_model():
    batch_size = 32
    img_size = 224

    # Data transformation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    full_dataset = datasets.ImageFolder(CONSOLIDATED_DATA_DIR, data_transforms['train'])
    
    # Train test split (80-20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = full_dataset.classes
    print("Classes:", class_names)
    
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transfer Learning with MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze base model layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    epochs = 5 # Small number of epochs for prototype
    best_acc = 0.0
    best_model_weights = model.state_dict()

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = model.state_dict()

    print(f'Training complete. Best Val Acc: {best_acc:4f}')
    
    # Save Model
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), 'psmear_model.pth')
    print("Model saved to psmear_model.pth")

if __name__ == "__main__":
    prepare_dataset()
    build_and_train_model()
