import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .context import CPM, AGCB_Element, AGCB_Patch
from .context_my import CPM_my, AGCB_Element, AGCB_Patch
from .fusion import *


__all__ = ['agpcnet']


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.5):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

def conv_1to3(img_gray):
    img = torch.cat((img_gray, img_gray, img_gray), dim=1)  # 在通道维度上进行拼接，得到[8, 3, 256, 256]
    return img

class AGPCNet(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch', gca_att='post',
                 drop=0.1):
        super(AGPCNet, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = conv_1to3(x)

        c1, c2, c3 = self.backbone(x)

        out = self.context(c3)

###############################################################################################################################
        # 输出特征图时注释
        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out.sigmoid()


class AGPCNet_Pro(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1):
        super(AGPCNet_Pro, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape

        c1, c2, c3 = self.backbone(x)

        out = self.context(c3)

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out.sigmoid()


def agpcnet(backbone, scales, reduce_ratios, gca_type, gca_att, drop):
    return AGPCNet(backbone=backbone, scales=scales, reduce_ratios=reduce_ratios, gca_type=gca_type, gca_att=gca_att, drop=drop)


#########################################################
###2.测试ASKCResUNet
import  numpy as np
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    layers = [3] * 3
    channels = [x * 1 for x in [8, 16, 32, 64]]
    in_channels = 3
    model= AGPCNet()

    model=model.cuda()
    # DATA = torch.randn(8,3,480,480).to(DEVICE)
    DATA = torch.randn(8, 1, 128, 128).to(DEVICE)

    output=model(DATA)
    print("output:",np.shape(output))
##########################################################

class AGPCNetmy(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch', gca_att='post',
                 drop=0.1):
        super(AGPCNetmy, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM_my(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = conv_1to3(x)

        c1, c2, c3 = self.backbone(x)

        out = self.context(c3)

###############################################################################################################################
        # 输出特征图时注释
        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out.sigmoid()