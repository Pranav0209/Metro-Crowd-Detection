import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load Dataset
train_dataset = CrowdHumanDataset(
    image_dir="/path/to/CrowdHuman/Images/Train",
    annotation_file="/path/to/CrowdHuman/annotation_train.odgt",
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 1)  # Regression output
resnet = resnet.to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.0001)

# Train Loop
for epoch in range(10):
    resnet.train()
    running_loss = 0.0
    for imgs, counts in train_loader:
        imgs = imgs.to(device)
        counts = counts.float().unsqueeze(1).to(device)

        outputs = resnet(imgs)
        loss = criterion(outputs, counts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/10] Loss: {running_loss/len(train_loader):.4f}")
