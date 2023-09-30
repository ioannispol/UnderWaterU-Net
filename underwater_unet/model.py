import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from underwater_unet.unet_blocks import (DoubleConv,
                                         DownConv,
                                         UpConv,
                                         OutConv,
                                         AttentionGate)


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


class AttentionUNet(UNet):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AttentionUNet, self).__init__(n_channels, n_classes, bilinear)
        
        # Define the attention gates
        self.att1 = AttentionGate(512, 512, 256)
        self.att2 = AttentionGate(256, 256, 128)
        self.att3 = AttentionGate(128, 128, 64)
        self.att4 = AttentionGate(64, 64, 32)

    def forward(self, x):
        # Encoder
        x1 = checkpoint(self.inc, x)
        x2 = checkpoint(self.down1, x1)
        x3 = checkpoint(self.down2, x2)
        x4 = checkpoint(self.down3, x3)
        x5 = checkpoint(self.down4, x4)
        
        # Apply attention before upsampling
        x4 = self.att1(g=x5, x=x4)
        x3 = self.att2(g=x4, x=x3)
        x2 = self.att3(g=x3, x=x2)
        x1 = self.att4(g=x2, x=x1)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits