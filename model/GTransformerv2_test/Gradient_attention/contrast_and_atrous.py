import torch.nn as nn
import torch.nn.functional as F
import torch

class Avg_ChannelAttention(nn.Module):
    def __init__(self, channels, r=4):
        super(Avg_ChannelAttention, self).__init__()
        self.avg_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 bz,C_out,h,w -> bz,C_out,1,1
            nn.Conv2d(channels, channels // r, 1, 1, 0),  # bz,C_out,1,1 -> bz,C_out/r,1,1
            nn.BatchNorm2d(channels // r),
            nn.ReLU(True),
            nn.Conv2d(channels // r, channels, 1, 1, 0),  # bz,C_out/r,1,1 -> bz,C_out,1,1
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.avg_channel(x)


class AttnContrastLayer(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=False):
        super(AttnContrastLayer, self).__init__()
        # 原始普通卷积
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                              padding='same', dilation=dilation, groups=groups, bias=bias)
        # 用于计算差分系数的全局注意力机制
        self.attn1 = Avg_ChannelAttention(channels)
        self.attn2 = Avg_ChannelAttention(channels)
    def forward(self, x,up):
        # 原始k*k滤波（卷积）
        out_normal = self.conv(x)
        # 系数
        theta = torch.sigmoid((self.attn1(x)+self.attn2(up))/2.)

        # 对k*k滤波器的权重求和，形成1*1滤波器进行滤波
        kernel_w1 = self.conv.weight.sum(2).sum(2)  # 对每一个k*k滤波器的权重求和 C_out,C_in
        kernel_w2 = kernel_w1[:, :, None, None]  # 扩充两个维度 C_out,C_in,1,1
        out_center = F.conv2d(input=x, weight=kernel_w2, bias=self.conv.bias, stride=self.conv.stride,
                              padding=0, groups=self.conv.groups)

        # 将k*k的滤波结果与1*1的滤波结果相减
        return theta * out_center - out_normal


class AtrousAttnWeight(nn.Module):
    def __init__(self, channels):
        super(AtrousAttnWeight, self).__init__()
        self.attn = Avg_ChannelAttention(channels)

    def forward(self, x):
        return self.attn(x)