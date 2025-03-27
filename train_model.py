

import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Training on {device}")

# Define training configurations
BATCH_SIZE = 8 
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5 

# Load the image processor (replaces ViTFeatureExtractor)
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Define dataset and transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 (ViT input size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

# Load dataset
train_dataset = datasets.ImageFolder(root="dataset/train/Images", transform=transform)
num_classes = len(train_dataset.classes)  # Dynamically detect number of breeds
print(f" Found {num_classes} classes (breeds)")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model with correct number of labels
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=num_classes,  # Dynamically set the number of breeds
    ignore_mismatched_sizes=True
).to(device)  # Move model to GPU (if available)

# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)  # Move data to GPU

        optimizer.zero_grad()
        outputs = model(images).logits  # ✅ Fix: Ensure images are in tensor format
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f" Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "dog_breed_model.pth")
print(" Model training complete! Model saved as 'dog_breed_model.pth'")
