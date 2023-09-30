import torch
import torch.nn.functional as F

from underwater_unet.unet_blocks import DoubleConv, DownConv, UpConv, OutConv


def test_DoubleConv():
    # Create a DoubleConv instance
    model = DoubleConv(in_channels=3, out_channels=64)

    # Generate a random tensor of size [batch_size, channels, height, width]
    x = torch.randn(1, 3, 64, 64)

    # Forward pass
    y = model(x)

    # Check the output dimensions
    assert y.shape == (1, 64, 64, 64)

    # Check if mid_channels defaults to out_channels when not provided
    assert model.double_conv[0].out_channels == 64

    # Create another instance with mid_channels specified
    model2 = DoubleConv(in_channels=3, out_channels=64, mid_channels=32)

    # Check if mid_channels is set correctly
    assert model2.double_conv[0].out_channels == 32


def test_DownConv():
    model = DownConv(in_channels=3, out_channels=64)
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, 64, 64, 64), f"Expected shape (1, 64, 64, 64) but got {y.shape}"
    assert isinstance(y, torch.Tensor), f"Expected output type torch.Tensor but got {type(y)}"


def test_UpConv():
    model = UpConv(in_channels=128, out_channels=64)

    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 64, 64)

    y = model(x1, x2)

    assert y.shape == (1, 64, 64, 64), f"Expected shape (1, 64, 64, 64) but got {y.shape}"

    assert isinstance(y, torch.Tensor), f"Expected output type torch.Tensor but got {type(y)}"

    upsampled_x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
    assert torch.allclose(upsampled_x1, model.up(x1), atol=1e-5), "Bilinear upsampling not applied correctly"


def test_OutConv():
    in_channels = 128
    out_channels = 64
    model = OutConv(in_channels=in_channels, out_channels=out_channels)

    x = torch.randn(1, in_channels, 64, 64)

    # Forward pass
    y = model(x)

    assert y.shape[1] == out_channels, f"Expected {out_channels} channels but got {y.shape[1]} channels"

    assert y.shape[2:] == x.shape[2:], f"Expected spatial dimensions {x.shape[2:]} but got {y.shape[2:]}"

    assert isinstance(y, torch.Tensor), f"Expected output type torch.Tensor but got {type(y)}"
