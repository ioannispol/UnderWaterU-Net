import torch

from underwater_unet.model import UNet


def test_UNet():
    # Create a UNet instance
    n_channels = 3
    n_classes = 2
    model = UNet(n_channels=n_channels, n_classes=n_classes)

    # Generate a random tensor of size [batch_size, channels, height, width]
    x = torch.randn(1, n_channels, 128, 128, requires_grad=True)

    # Forward pass
    y = model(x)

    # Check if the tensor has been transformed correctly
    assert y.shape == (1, n_classes, 128, 128), f"Expected shape (1, {n_classes}, 128, 128) but got {y.shape}"

    # Ensure the output is of the correct type
    assert isinstance(y, torch.Tensor), f"Expected output type torch.Tensor but got {type(y)}"

    # Ensure the values are within a reasonable range (sanity check, not strictly required)
    assert y.max() <= 10 and y.min() >= -10, "Output values are out of a reasonable range"
