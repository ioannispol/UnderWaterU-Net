import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from underwater_unet.model import UNet
from utils.data_load import UnderwaterDataset

# Hyperparameters and setup
num_epochs = 10
learning_rate = 0.001
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transform
transform = transforms.Compose([transforms.ToTensor()])

# Load dataset and dataloader
train_dataset = UnderwaterDataset(images_dir='data/images', mask_dir='data/masks', resize_to=None)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss and optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    os.makedirs("experiment", exist_ok=True)
    model.train()
    for batch in train_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).unsqueeze(1).float()  # Add a channel dimension and convert to float

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), f"experiment/model_epoch_{epoch + 1}.pth")

print("Training completed.")
