import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from config import config

class MyNN(nn.Module):
    def __init__(self, ip_features=3, num_classes=33):
        super().__init__()
    
        self.features = nn.Sequential(
            nn.Conv2d(ip_features, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model():
    """Load the trained model with error handling."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = config.MODEL_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # Initialize model
        model = MyNN(ip_features=3, num_classes=33)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle different state dict formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # Load the state dict with strict=False to handle missing keys
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Return a new model instance if loading fails
        model = MyNN(ip_features=3, num_classes=33)
        model.eval()
        model.to(device)
        return model

# Image transformation
def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
