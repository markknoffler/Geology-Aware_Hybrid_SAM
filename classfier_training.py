

import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import shutil  # Added for file operations
import csv  # Added for CSV logging

# -----------------------------
# Step 1: Define Improved CNN Model with Attention
# -----------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    """Convolutional block with BN and SE attention"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.se(x)
        return x

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ImprovedCNN, self).__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Additional convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.se4 = SEBlock(256)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # Adjusted for new spatial dimensions
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Input: 3x128x128
        x = self.pool(self.block1(x))  # 32x64x64
        x = self.pool(self.block2(x))  # 64x32x32
        x = self.pool(self.block3(x))  # 128x16x16
        
        # Additional conv block
        x = F.relu(self.bn4(self.conv4(x)))  # 256x16x16
        x = self.se4(x)
        x = self.pool(x)  # 256x8x8
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        return x

# -----------------------------
# Step 2: Create Custom Dataset (Unchanged)
# -----------------------------
class LandslideDataset(Dataset):
    def __init__(self, landslide_dir, non_landslide_dir, transform=None):
        self.transform = transform
        self.data = []

        for img_name in os.listdir(landslide_dir):
            img_path = os.path.join(landslide_dir, img_name)
            if os.path.isfile(img_path):
                self.data.append((img_path, 1))  # Class 1

        for img_name in os.listdir(non_landslide_dir):
            img_path = os.path.join(non_landslide_dir, img_name)
            if os.path.isfile(img_path):
                self.data.append((img_path, 0))  # Class 0

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# Step 3: Data Preparation (Unchanged)
# -----------------------------

# Placeholder directories for training data
landslide_dir = '/home/user5006/Documents/SAM/segment_anything/segment-anything/color_padded/image'
non_landslide_dir = '/home/user5006/Documents/SAM/segment_anything/segment-anything/color_padded/padded_non_landslide/padded'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

full_dataset = LandslideDataset(landslide_dir, non_landslide_dir, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Step 4: Model Training with Checkpoints and CSV Logging
# -----------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ADD YOUR CHECKPOINT DIRECTORY HERE
checkpoint_dir = "/home/user5006/Documents/SAM/segment_anything/segment-anything/classifier_results"  # <<< UPDATE THIS PATH

# Create directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

# Create CSV file path
csv_path = os.path.join(checkpoint_dir, 'training_results.csv')

# Write CSV header
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_acc', 'lr'])

model = ImprovedCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Added weight decay

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step(val_acc)  # Update learning rate
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save metrics to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_loss, train_acc, val_acc, current_lr])
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, os.path.join(checkpoint_dir, 'best_model.pth'))
    
    # Save regular checkpoint
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.6f}')

print(f'Training complete. Best validation accuracy: {best_val_acc:.4f}')

# =================================================================
# NEW SECTION: Classify images from source directories
# =================================================================


# Placeholder directories for classification
source_dirs = [
    "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/Kameng_River/512x512",   # <<< UPDATE THESE PATHS
    "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/tupul_noney_manipur/3-4km_land",   # <<< UPDATE THESE PATHS
    "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/Mowii–Senki/512x512"    # <<< UPDATE THESE PATHS
]

landslide_output_dir = "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/landslide"          # <<< UPDATE THIS PATH
non_landslide_output_dir = "/home/user5006/Documents/SAM/segment_anything/segment-anything/NE_dataset/non-landslide"  # <<< UPDATE THIS PATH


# Create output directories if they don't exist
os.makedirs(landslide_output_dir, exist_ok=True)
os.makedirs(non_landslide_output_dir, exist_ok=True)

# Load the best model for inference
checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

# Define the same transform used during validation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Classify images from all source directories
for source_dir in source_dirs:
    for img_name in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_name)
        
        # Skip directories
        if not os.path.isfile(img_path):
            continue
            
        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()
                
            # Copy image to appropriate directory
            if prediction == 1:  # Landslide
                dest_path = os.path.join(landslide_output_dir, img_name)
            else:  # Non-landslide
                dest_path = os.path.join(non_landslide_output_dir, img_name)
                
            shutil.copy2(img_path, dest_path)
            print(f"Copied {img_path} to {dest_path}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

print("Classification and sorting completed!")