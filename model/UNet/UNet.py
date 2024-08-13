import torch
import torch.nn as nn
from torch.nn import Flatten
import torch.nn.functional as F
# from .CDCNs.Gradient_model import ExpansionContrastModule
# from .CDCNs.CDCN import Conv2d_cd
# from model.utils import init_weights, count_param
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        x = torch.cat([skip_x, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class UNet(nn.Module):
    def __init__(self,  n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 16  # basic channel 64
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)  # 64  128
        self.encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1)  # 64  128
        self.encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  # 64  128
        self.encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  # 64  128
        self.decoder4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.decoder3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.decoder2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.decoder1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.encoder1(self.pool(x1))
        x3 = self.encoder2(self.pool(x2))
        x4 = self.encoder3(self.pool(x3))
        d5 = self.encoder4(self.pool(x4))
        d4 = self.decoder4(d5, x4)
        d3 = self.decoder3(d4, x3)
        d2 = self.decoder2(d3, x2)
        out = self.outc(self.decoder1(d2, x1))
        return out.sigmoid()
