import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from underwater_unet.model import UNet

# Define the dataset
class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.image_files = os.listdir(self.images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.images_dir, img_name))
        mask = Image.open(os.path.join(self.masks_dir, img_name))
        sample = {'image': image, 'mask': mask}
        
        if self.transform:
            sample['image'] = self.transform(image)
            sample['mask'] = self.transform(mask)

        return sample

# Hyperparameters and setup
num_epochs = 10
learning_rate = 0.001
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset and dataloader
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = UnderwaterDataset(root_dir='data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss and optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
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
