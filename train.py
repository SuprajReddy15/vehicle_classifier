import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn, optim
from sklearn.metrics import classification_report
import numpy as np

# Paths
dataset_dir = r'C:\Users\Supra\OneDrive\Desktop\vehicle_jod\vehicle\images'
model_save_path = r'/model/vehicle_classification_model.pth'

# ✅ Safety check
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset.classes))
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# Save model
torch.save(model.state_dict(), model_save_path)
print(f"\n✅ Model saved to: {model_save_path}")

# Evaluation
print("\n--- Evaluation Report ---")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification Report
print(classification_report(all_labels, all_preds, target_names=dataset.classes))
