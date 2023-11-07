import os
import argparse


import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from underwater_unet.model import UNet
from utils.data_load import UnderwaterDataset
from utils.dice_score import dice_loss


def train_model(
        model,
        device,
        epochs=10,
        batch_size=1,
        learning_rate=0.001,
        val_percent=10.0,
        save_checkpoint=True,
        weight_decay=1e-8,
        momentum=0.9,
        amp=False,  # Add AMP flag to the function arguments
        gradient_clipping=1.0
):
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any other transformations here
    ])

    # Load dataset and dataloader
    train_dataset = UnderwaterDataset(images_dir='data/images', mask_dir='data/masks', augmentations=transform)
    n_val = int(len(train_dataset) * val_percent / 100)
    n_train = len(train_dataset) - n_val
    train_set, val_set = random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss, optimizer, and AMP scaler
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(momentum, 0.999))
    scaler = GradScaler(enabled=amp)

    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Initialize experimnet logging
    wandb.init(project="UW-Unet", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_percent": val_percent,
        "save_checkpoint": save_checkpoint,
        "amp": amp,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "gradient_clipping": gradient_clipping,
    })
    config = wandb.config

    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0
        # Inside the training loop
        for batch in tqdm(train_loader, desc="Training", leave=False):
            images = batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = batch['mask'].to(device=device, dtype=torch.float32 if model.n_classes == 1 else torch.long)

            with autocast(enabled=amp):
                masks_pred = model(images)
                if model.n_classes == 1:
                    masks_pred = masks_pred.squeeze(1)  # Remove channel dim because we have a single class
                    masks = masks.squeeze(1)  # Squeeze the masks for binary case
                    loss = criterion(masks_pred, masks)
                    loss += dice_loss(torch.sigmoid(masks_pred), masks, multiclass=False)
                else:
                    # No need to squeeze masks for multi-class since CrossEntropy expects a 1D target
                    loss = criterion(masks_pred, masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1),
                        F.one_hot(masks, model.n_classes).permute(0, 3, 1, 2),
                        multiclass=True
                    )
            # Backward and optimization steps remain the same

            assert masks_pred.shape == \
                masks.shape, f"Output shape {masks_pred.shape} and mask shape {masks.shape} do not match"
            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping
            if gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Log training loss after each epoch
        train_loss = epoch_loss / len(train_loader)
        wandb.log({
            "epoch": epoch,
            "training_loss": train_loss,
        }, step=epoch)


        val_loss = 0
        model.eval()

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device).unsqueeze(1).float()
                outputs = model(images)
                probs = torch.sigmoid(outputs)  # Apply sigmoid for dice_loss
                dice = dice_loss(probs, masks, multiclass=False)
                bce = criterion(outputs, masks)
                loss = bce + dice  # Consistent with training loss
                val_loss += loss.item()

                # Log the predicted masks and the true masks as images
                wandb.log({
                    "val_images": [wandb.Image(image.cpu(), caption="Val Image") for image in images],
                    "val_masks": [wandb.Image(mask.cpu(), caption="True Mask") for mask in masks],
                    "val_predictions": [wandb.Image(torch.sigmoid(output).cpu(),
                                                    caption="Pred Mask") for output in outputs],
                }, step=epoch)

        scheduler.step(val_loss)
        # Calculate and log the average validation loss
        val_loss_avg = val_loss / len(val_loader)
        wandb.log({
            "epoch": epoch,
            "validation_loss": val_loss_avg,
        }, step=epoch)

        # Log histograms of model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                wandb.log({f"{name}": wandb.Histogram(param.detach().cpu().numpy())}, step=epoch)

        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}, \
              Validation Loss: {val_loss / len(val_loader):.4f}')

        if config.save_checkpoint:
            checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)  # Make sure the checkpoint directory exists
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            wandb.save(checkpoint_path)
            wandb.log({"checkpoints": wandb.save(checkpoint_path)})

    print("Training completed.")

        scheduler.step(val_loss)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val-percent', '-v', dest='val_percent', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--save-checkpoint', action='store_true', help='Save a checkpoint at the end of each epoch')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision for training')
    parser.add_argument('--weight-decay', type=float, default=1e-8, help='Weight decay used in optimization')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum used in optimization')
    parser.add_argument('--n-classes', type=int, default=1, help='Number of classes for segmentation')
    parser.add_argument('--bilinear', action='store_true', help='Use bilinear upsampling')
    parser.add_argument('--gradient-clipping', type=float, default=1.0, help='Max norm of the gradients')
    # You can add more arguments as required

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = get_args()

    # Initialize wandb
    wandb.init(project="UW-Unet1", config=args)

    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)
    train_model(model=model, device=device)

    # Initialize and transfer the U-Net model to the device
    model = UNet(n_channels=3, n_classes=args.n_classes, bilinear=args.bilinear).to(device)

    # Pass the configurations from wandb to your training function
    train_model(
        model=model,
        device=device,
        epochs=wandb.config.epochs,
        batch_size=wandb.config.batch_size,
        learning_rate=wandb.config.learning_rate,
        val_percent=wandb.config.val_percent,
        save_checkpoint=wandb.config.save_checkpoint,
        amp=wandb.config.amp,
        weight_decay=wandb.config.weight_decay,
        momentum=wandb.config.momentum,
        gradient_clipping=wandb.config.gradient_clipping
    )
