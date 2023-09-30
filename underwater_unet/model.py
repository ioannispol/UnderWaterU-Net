import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from underwater_unet.unet_blocks import (DoubleConv,
                                         DownConv,
                                         UpConv,
                                         OutConv)


class UNet(nn.Module):

    def __init__(self, n_channels: int, n_classes, bilinear=False) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (DownConv(64, 128))
        self.down2 = (DownConv(128, 256))
        self.down3 = (DownConv(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (DownConv(512, 1024 // factor))
        self.up1 = (UpConv(1024, 512 // factor, bilinear))
        self.up2 = (UpConv(512, 256 // factor, bilinear))
        self.up3 = (UpConv(256, 128 // factor, bilinear))
        self.up4 = (UpConv(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = checkpoint(self.inc, x)
        x2 = checkpoint(self.down1, x1)
        x3 = checkpoint(self.down2, x2)
        x4 = checkpoint(self.down3, x3)
        x5 = checkpoint(self.down4, x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
