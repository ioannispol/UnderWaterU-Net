import torch
import torchviz
from torch.autograd import Variable

from underwater_unet.model import AttentionUNet

# Assuming the AttentionUNet model is defined in the script
model = AttentionUNet(n_channels=3, n_classes=1)

dummy_input = Variable(torch.randn(1, 3, 256, 256))
dot = torchviz.make_dot(model(dummy_input), params=dict(model.named_parameters()))

dot.format = 'png'
dot.render("unet_attention_model.png")