from torch import nn
import torch
from torch.nn import functional as F
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, x):
        x1 = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(x, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class RepBlock_fuse(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels
        self.layernorm = LayerNorm(in_channels,eps=1e-6)
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels,in_channels,kernel_size=5,padding=2,groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,3),padding=(0,1),groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(3,1),padding=(1,0),groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,5),padding=(0,2),groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(5,1),padding=(2,0),groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,7),padding=(0,3),groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(7,1),padding=(3,0),groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels*2,out_channels,kernel_size=3,padding=1,stride=1),
                                      nn.BatchNorm2d(out_channels),
                                      nn.GELU())
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels,out_channels=in_channels,kernel_size=2,stride=2,padding=0)

    def forward(self,high,low):
        
        channel_att_vec = self.ca(high)
        low_out = channel_att_vec * low

        high = self.upsample(high)
        x_init = self.dconv5_5(low)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        high_out = spatial_att * high
        out = torch.concat([high_out,low_out],dim=1)
        out = self.out_conv(out)
        return out
class RepBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels
        # assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1,groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,5),padding=(0,2),groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(5,1),padding=(2,0),groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,7),padding=(0,3),groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(7,1),padding=(3,0),groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,9),padding=(0,4),groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(9,1),padding=(4,0),groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()
    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out
class ChannelAttention_fuse(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention_fuse, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, low,high):
        x1 = F.adaptive_avg_pool2d(high, output_size=(1, 1))
        # print('x:', x1.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(low, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x