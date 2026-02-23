import torch
import torch.nn as nn

# Define the CNN model architecture
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=38):  # Default to 38 classes, adjust as needed
        super(PlantDiseaseCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Forward pass
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.global_avg_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def load_model(model_path='plant_cnn_model.pt', num_classes=38, device='cpu'):
    """
    Load the trained plant disease model
    """
    try:
        # Try to load the entire model
        model = torch.load(model_path, map_location=device)
        print(f"Model loaded from {model_path}")
        return model
    except:
        # If that fails, try loading state dict
        print("Loading state dict...")
        model = PlantDiseaseCNN(num_classes=num_classes)
        
        # Check if file contains state_dict or the entire model
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            # Try to use it directly
            return checkpoint
            
        model.eval()
        return model

def preprocess_image(image, img_size=(256, 256)):
    """
    Preprocess image for model input
    """
    import cv2
    from PIL import Image
    import numpy as np
    
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
        
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Resize
    pil_image = pil_image.resize(img_size)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(np.array(pil_image)).float()
    
    # Normalize (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Permute dimensions and normalize
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        image_tensor = (image_tensor - mean) / std
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor