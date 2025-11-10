#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

class DexHandDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load dataset
        with open(os.path.join(data_dir, 'dataset.json'), 'r') as f:
            self.data = json.load(f)
        
        # Map gestures to IDs
        self.gesture_map = {
            'reset': 0, 'fist': 1, 'point': 2, 
            'peace': 3, 'open_hand': 4, 'wave': 5
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Load image
        img_path = sample['image_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get gesture label
        gesture_id = self.gesture_map.get(sample['gesture'], 0)
        
        return {
            'image': image,
            'label': torch.tensor(gesture_id, dtype=torch.long)
        }

class PolicyNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Simple CNN for gesture classification
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

def train(data_dir='demonstrations', epochs=50, batch_size=8):
    """Train a policy from collected demonstrations"""
    
    # Data transforms
    transform = torch.nn.Sequential(
        torch.nn.RandomRotation(15),
        torch.nn.ColorJitter(0.2, 0.2, 0.2),
        torch.nn.Resize((224, 224)),
        torch.nn.Normalize(mean=[0.5, 0.5, 0.5], 
                          std=[0.5, 0.5, 0.5])
    )
    
    # Dataset
    dataset = DexHandDataset(data_dir, transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = PolicyNet(num_classes=len(dataset.gesture_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch['image'])
            loss = criterion(outputs, batch['label'])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'dexhand_policy.pth')
    print("Training complete! Model saved to dexhand_policy.pth")

if __name__ == '__main__':
    train()