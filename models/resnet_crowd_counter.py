import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np

class ResNet18CrowdCounter(nn.Module):
    def __init__(self, load_weights=True):
        super(ResNet18CrowdCounter, self).__init__()
        # Load pretrained ResNet18 backbone
        resnet18 = models.resnet18(pretrained=load_weights)
        
        # Modify the first conv layer to accept grayscale input (1 channel)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy weights from pretrained model for all layers except first conv
        if load_weights:
            self.conv1.weight.data = torch.mean(resnet18.conv1.weight.data, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        
        # Use all layers from ResNet except the final FC layer
        self.features = nn.Sequential(
            self.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool
        )
        
        # Custom regression head for crowd counting
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regression_head(x)
        return x.squeeze()


class CrowdCounter:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, load_weights=True):
        """Load the ResNet18 crowd counter model
        
        Args:
            load_weights: Whether to load pretrained weights
            
        Returns:
            model: Loaded model
        """
        if self.model is not None:
            return self.model
        
        # Initialize model
        self.model = ResNet18CrowdCounter(load_weights=load_weights)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move model to GPU if available
        self.model = self.model.to(self.device)
        
        return self.model
    
    def predict_crowd(self, img_path):
        """Predict crowd count from an image
        
        Args:
            img_path: Path to the input image
            
        Returns:
            count: Predicted crowd count
            heatmap: Visualization of the crowd density
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            return 0, None
        
        # Resize image to model input size
        img_resized = cv2.resize(img, (224, 224))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            count = self.model(img_tensor).item()
        
        # Round to nearest integer
        count = round(count)
        
        # Create a simple heatmap for visualization
        heatmap = self._create_heatmap(img)
        
        return count, heatmap
    
    def _create_heatmap(self, img):
        """Create a simple heatmap for visualization
        
        Args:
            img: Input image
            
        Returns:
            heatmap: Visualization of the crowd density
        """
        # This is a placeholder for a more sophisticated heatmap generation
        # In a real implementation, you would use the model's features to create a density map
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Apply threshold to create a binary image
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        
        # Apply color map
        heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        
        return overlay