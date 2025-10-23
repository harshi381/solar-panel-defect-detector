-> Dataset Description:-
Solar Panel Images Clean and Faulty Images 
Sourced from Kaggle and then augemented as per Requirements:-
The dataset contains RGB images of solar panels captured using drones (UAVs) under real-world lighting and weather conditions.
Each image is labeled according to the type of fault or soiling present on the panel.
Images were preprocessed and resized to 224 × 224 pixels for compatibility with the ResNet model.
The dataset includes Six distinct classes:
Bird-Drop
Dusty 
Clean
Electrical Damage
Snow Covered
Physical Damage


The model is using pretrained Resnet. The training code is as below :-

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

# === 1. CONFIG ===
data_dir = '/kaggle/input/augmenteddataset/AugmentedDataset'
batch_size = 32
num_epochs = 50
patience = 5
lr = 1e-3
input_size = 224
num_classes = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Dataset ===
class CancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_paths)

# === 3. Stratified Split ===
def get_dataset_split(data_dir, test_size=0.2):
    class_names = sorted(os.listdir(data_dir))
    all_paths, all_labels = [], []

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            all_paths.append(os.path.join(class_path, img_name))
            all_labels.append(idx)

    all_paths, all_labels = np.array(all_paths), np.array(all_labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, val_idx = next(sss.split(all_paths, all_labels))
    return (all_paths[train_idx], all_labels[train_idx]), (all_paths[val_idx], all_labels[val_idx]), class_names

(train_paths, train_labels), (val_paths, val_labels), class_names = get_dataset_split(data_dir)

# === 4. Transforms ===
train_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),  # slightly random zoom
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # new line
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = CancerDataset(train_paths, train_labels, transform=train_transform)
val_dataset = CancerDataset(val_paths, val_labels, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# === 5. Model ===
model = models.resnet18(pretrained=True)

# Add a small dropout before the final FC layer (no other changes)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

# === 6. Training Tools ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# === 7. Training Loop with Early Stopping ===
best_val_loss = np.inf
best_model_path = 'best_model.pth'
early_stopping_counter = 0
train_losses, val_losses, val_accuracies, train_accuracies = [], [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct_train, total_train = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # === Validation ===
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

# === 8. Load Best Model & Evaluate ===
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=class_names))

# === 9. Plot Metrics ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
