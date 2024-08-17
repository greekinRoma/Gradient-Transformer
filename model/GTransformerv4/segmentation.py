import torch
import torch.nn as nn
from torch.nn import Flatten
import torch.nn.functional as F
from .Gradient_attention.contrast_and_atrous import AttnContrastLayer
from .CDCNs.Gradient_model import ExpansionContrastModule
from .AttentionModule import *
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='LeakyReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
def _make_nConv(in_channels, out_channels, nb_Conv, activation='LeakyReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)
class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='LeakyReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        # self.cattn = ChannelAttention(input_channels=in_channels//2,internal_neurons=in_channels//16)
        self.cattn = eca_layer_fuse(channel=in_channels//2)
        # self.sattn = EMA_fuse(channels=in_channels//2)
    def forward(self,d,c,xin):
        d = self.cattn(low=xin,high=d)
        d = self.up(d)
        # x = self.sattn(low=d,high=xin)
        x = torch.cat([c, d], dim=1)  # dim 1 is the channel dimension
        x = self.nConvs(x)
        return x
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

class GTransformerv4(nn.Module):
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
        self.encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)  
        self.encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1) 
        self.encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  
        self.encoder4 = self._make_layer(block, in_channels * 8, in_channels * 8, 1)  
        self.contras1 = ExpansionContrastModule(in_channels=in_channels*1,out_channels=in_channels*1,width=img_size//1,height=img_size//1,shifts=[1,3])
        self.contras2 = ExpansionContrastModule(in_channels=in_channels*2,out_channels=in_channels*2,width=img_size//2,height=img_size//2,shifts=[1,3])
        self.contras3 = ExpansionContrastModule(in_channels=in_channels*4,out_channels=in_channels*4,width=img_size//4,height=img_size//4,shifts=[1,3])
        self.contras4 = ExpansionContrastModule(in_channels=in_channels*8,out_channels=in_channels*8,width=img_size//8,height=img_size//8,shifts=[1,3])
        self.decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))


    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #encoder
        x1 = self.inc(x)  # 64 224 224
        x2 = self.encoder1(self.pool(x1))  # 128 112 112
        x3 = self.encoder2(self.pool(x2))  # 256 56  56
        x4 = self.encoder3(self.pool(x3))  # 512 28  28
        d5 = self.encoder4(self.pool(x4))  # 512 14  14
        # Transfor_layer
        c1 = self.contras1(x1)
        c2 = self.contras2(x2)
        c3 = self.contras3(x3)
        c4 = self.contras4(x4)
        # decoder
        d4 = self.decoder4(d5, c4, x4)
        d3 = self.decoder3(d4, c3, x3)
        d2 = self.decoder2(d3, c2, x2)
        out = self.outc(self.decoder1(d2, c1, x1))
        return out.sigmoid()