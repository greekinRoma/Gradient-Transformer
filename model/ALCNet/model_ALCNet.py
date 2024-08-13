import torch
import torch.nn as nn
import torch.nn.functional as F


# 等价于 gluoncv.model_zoo.cifarresnet 中的 CIFARBasicBlockV1 (即残差单元)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 为了进行短接，需要通过1x1卷积将输入输出调整为一致大小
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.shortcut:
            residual = self.shortcut(residual)

        out = F.relu(x + residual, True)
        return out


# 等价于 gluoncv.model_zoo.fcn 中的 _FCNHead (即全连接分割头)
class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class BiGlobal_MPCMFuse(nn.Module):
    def __init__(self, channels=64):
        super(BiGlobal_MPCMFuse, self).__init__()
        inter_channels = int(channels // 2)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

        self.topdown_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.bottomup_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, cen):
        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)
        pcm13 = self.bn1(pcm13)
        pcm17 = self.bn2(pcm17)

        topdown_wei = self.topdown_att(pcm17)
        bottomup_wei = self.bottomup_att(pcm13)

        xo = topdown_wei * pcm13 + bottomup_wei * pcm17

        return xo


class BiLocal_MPCMFuse(nn.Module):
    def __init__(self, channels=64):
        super(BiLocal_MPCMFuse, self).__init__()
        inter_channels = int(channels // 2)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

        self.topdown_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        self.bottomup_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, cen):
        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)
        pcm13 = self.bn1(pcm13)
        pcm17 = self.bn2(pcm17)

        topdown_wei = self.topdown_att(pcm17)
        bottomup_wei = self.bottomup_att(pcm13)

        xo = topdown_wei * pcm13 + bottomup_wei * pcm17

        return xo


class Add_MPCMFuse(nn.Module):
    def __init__(self, channels=64):
        super(Add_MPCMFuse, self).__init__()

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, cen):
        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)
        pcm13 = self.bn1(pcm13)
        pcm17 = self.bn2(pcm17)

        xo = pcm13 + pcm17

        return xo


class GlobalSK_MPCMFuse(nn.Module):
    def __init__(self, channels=64):
        super(GlobalSK_MPCMFuse, self).__init__()
        inter_channels = int(channels // 2)

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, cen):
        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)

        xa = pcm13 + pcm17
        wei = self.global_att(xa)

        xo = 2 * pcm13 * wei + 2 * pcm17 * (1 - wei)

        return xo


class LocalSK_MPCMFuse(nn.Module):
    def __init__(self, channels=64):
        super(LocalSK_MPCMFuse, self).__init__()
        inter_channels = int(channels // 2)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, cen):
        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)

        xa = pcm13 + pcm17
        wei = self.local_att(xa)

        xo = 2 * pcm13 * wei + 2 * pcm17 * (1 - wei)

        return xo


class BottomUpLocal_FPNFuse(nn.Module):
    def __init__(self, channels=64):
        super(BottomUpLocal_FPNFuse, self).__init__()
        inter_channels = int(channels // 1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

        self.bottomup_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, 1, 1, 0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),

            nn.Conv2d(inter_channels, channels, 1, 1, 0),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, x, residual):
        x = self.bn1(x)
        residual = self.bn2(residual)

        bottomup_wei = self.bottomup_att(residual)

        xo = bottomup_wei * x + residual

        return xo


class MPCMResNetFPN(nn.Module):
    def __init__(self, layers= [4] * 3 , channels=[8, 16, 32, 64], shift=3, r=2, scale_mode='localsk', pyramid_fuse='bottomuplocal'):
        super(MPCMResNetFPN, self).__init__()

        # layers = [4] * 3  # 4
        # channels = [8, 16, 32, 64]
        self.layer_num = len(layers)  # 层数 4

        self.r = r
        self.shift = shift
        self.scale_mode = scale_mode
        self.pyramid_fuse = pyramid_fuse

        # bz,3,256,256 -> bz,16,256,256
        stem_width = channels[0]
        self.stem = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, stem_width * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width * 2),
            nn.ReLU(True),
        )

        # bz,16,256,256 -> bz,16,256,256
        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layers[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        # bz,16,256,256 -> bz,32,128,128
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layers[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        # bz,32,128,128 -> bz,64,64,64
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layers[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        # c3: bz,64,64,64 -> bz,64,64,64
        # c2: bz,32,128,128 -> bz,32,128,128
        # c1: bz,16,256,256 -> bz,16,256,256
        if self.scale_mode == 'biglobal':
            self.fuse_mpcm_c1 = BiGlobal_MPCMFuse(channels=channels[1])
            self.fuse_mpcm_c2 = BiGlobal_MPCMFuse(channels=channels[2])
            self.fuse_mpcm_c3 = BiGlobal_MPCMFuse(channels=channels[3])
        elif self.scale_mode == 'bilocal':
            self.fuse_mpcm_c1 = BiLocal_MPCMFuse(channels=channels[1])
            self.fuse_mpcm_c2 = BiLocal_MPCMFuse(channels=channels[2])
            self.fuse_mpcm_c3 = BiLocal_MPCMFuse(channels=channels[3])
        elif self.scale_mode == 'add':
            self.fuse_mpcm_c1 = Add_MPCMFuse(channels=channels[1])
            self.fuse_mpcm_c2 = Add_MPCMFuse(channels=channels[2])
            self.fuse_mpcm_c3 = Add_MPCMFuse(channels=channels[3])
        elif self.scale_mode == 'globalsk':
            self.fuse_mpcm_c1 = GlobalSK_MPCMFuse(channels=channels[1])
            self.fuse_mpcm_c2 = GlobalSK_MPCMFuse(channels=channels[2])
            self.fuse_mpcm_c3 = GlobalSK_MPCMFuse(channels=channels[3])
        elif self.scale_mode == 'localsk':
            self.fuse_mpcm_c1 = LocalSK_MPCMFuse(channels=channels[1])
            self.fuse_mpcm_c2 = LocalSK_MPCMFuse(channels=channels[2])
            self.fuse_mpcm_c3 = LocalSK_MPCMFuse(channels=channels[3])

        # bz,64,64,64 -> bz,16,64,64
        self.dec_c3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(True),
        )

        # bz,32,128,128 -> bz,16,128,128
        self.dec_c2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(True),
        )

        self.bottomuplocal_fpn_2 = BottomUpLocal_FPNFuse(channels=channels[1])
        self.bottomuplocal_fpn_1 = BottomUpLocal_FPNFuse(channels=channels[1])

        self.head = _FCNHead(channels[1], 1)

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        # 仅第一个block单元负责增加通道数和降采样，后续block单元都保持Tensor形状不变
        layer = [block(in_channels, out_channels, stride)]
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layer)

    def forward(self, x):
        # FPN模块
        # bz,3,256,256 -> bz,64,64,64
        _, _, orig_hei, orig_wid = x.shape
        x = self.stem(x)  # bz,16,256,256
        c1 = self.layer1(x)  # bz,16,256,256
        _, _, c1_hei, c1_wid = c1.shape  # 256,256
        c2 = self.layer2(c1)  # bz,32,128,128
        _, _, c2_hei, c2_wid = c2.shape  # 128,128
        c3 = self.layer3(c2)  # bz,64,64,64
        _, _, c3_hei, c3_wid = c3.shape  # 64,64

        # ------由第3阶特征计算第3阶对比度特征，并通过上采样调整至统一大小------
        # bz,64,64,64 -> bz,64,64,64  (MPCM)
        if self.scale_mode == 'Single':  # 单一尺度PCM
            c3pcm = cal_pcm(c3, shift=self.shift)
        elif self.scale_mode == 'Multiple':  # 多尺度PCM(13,17)
            c3pcm = self.cal_mpcm(c3)
        elif self.scale_mode == 'biglobal':
            c3pcm = self.fuse_mpcm_c3(c3)
        elif self.scale_mode == 'bilocal':
            c3pcm = self.fuse_mpcm_c3(c3)
        elif self.scale_mode == 'add':
            c3pcm = self.fuse_mpcm_c3(c3)
        elif self.scale_mode == 'globalsk':
            c3pcm = self.fuse_mpcm_c3(c3)
        elif self.scale_mode == 'localsk':
            c3pcm = self.fuse_mpcm_c3(c3)
        else:
            raise ValueError("unknow self.scale_mode")
        # bz,64,64,64 -> bz,16,64,64  (统一通道数)
        c3pcm = self.dec_c3(c3pcm)
        # bz,16,64,64 -> bz,16,128,128  (统一尺寸)
        up_c3pcm = F.interpolate(c3pcm, size=[c2_hei, c2_wid], mode='bilinear')
        # -------------------------第3阶特征-------------------------

        # ------由第2阶特征计算第2阶对比度特征，并与第3阶特征融合，上采样------
        # bz,32,128,128 -> bz,32,128,128  (MPCM)
        if self.scale_mode == 'Single':
            c2pcm = cal_pcm(c2, shift=self.shift)
        elif self.scale_mode == 'Multiple':
            c2pcm = self.cal_mpcm(c2)
        elif self.scale_mode == 'biglobal':
            c2pcm = self.fuse_mpcm_c2(c2)
        elif self.scale_mode == 'bilocal':
            c2pcm = self.fuse_mpcm_c2(c2)
        elif self.scale_mode == 'add':
            c2pcm = self.fuse_mpcm_c2(c2)
        elif self.scale_mode == 'globalsk':
            c2pcm = self.fuse_mpcm_c2(c2)
        elif self.scale_mode == 'localsk':
            c2pcm = self.fuse_mpcm_c2(c2)
        else:
            raise ValueError("unknow self.scale_mode")
        # bz,32,128,128 -> bz,16,128,128  (统一通道数)
        c2pcm = self.dec_c2(c2pcm)
        # bz,16,128,128  (特征融合)
        c23pcm = self.bottomuplocal_fpn_2(up_c3pcm, c2pcm)
        # bz,16,128,128 -> bz,16,256,256  (统一尺寸)
        up_c23pcm = F.interpolate(c23pcm, size=[c1_hei, c1_wid], mode='bilinear')
        # -------------------------第2阶特征-------------------------

        # ---------由第1阶特征计算第1阶对比度特征，并与第2阶特征融合---------
        # bz,16,256,256 -> bz,16,256,256  (MPCM)
        if self.scale_mode == 'Single':
            c1pcm = cal_pcm(c1, shift=self.shift)
        elif self.scale_mode == 'Multiple':
            c1pcm = self.cal_mpcm(c1)
        elif self.scale_mode == 'biglobal':
            c1pcm = self.fuse_mpcm_c1(c1)
        elif self.scale_mode == 'bilocal':
            c1pcm = self.fuse_mpcm_c1(c1)
        elif self.scale_mode == 'add':
            c1pcm = self.fuse_mpcm_c1(c1)
        elif self.scale_mode == 'globalsk':
            c1pcm = self.fuse_mpcm_c1(c1)
        elif self.scale_mode == 'localsk':
            c1pcm = self.fuse_mpcm_c1(c1)
        else:
            raise ValueError("unknow self.scale_mode")
        # bz,16,256,256  (特征融合)
        out = self.bottomuplocal_fpn_1(up_c23pcm, c1pcm)

        pred = self.head(out)
        out = F.interpolate(pred, size=[orig_hei, orig_wid], mode='bilinear')

        return out.sigmoid()

    def cal_mpcm(self, cen):
        # pcm11 = cal_pcm(cen, shift=11)
        pcm13 = cal_pcm(cen, shift=13)
        pcm17 = cal_pcm(cen, shift=17)
        mpcm = torch.maximum(pcm13, pcm17)
        # mpcm = torch.maximum(pcm11, nd.maximum(pcm13, pcm17))

        return mpcm


def circ_shift(cen, shift):
    _, _, hei, wid = cen.shape

    ######## B1 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B1_NW = cen[:, :, shift:, shift:]  # B1_NW is cen's SE
    B1_NE = cen[:, :, shift:, :shift]  # B1_NE is cen's SW
    B1_SW = cen[:, :, :shift, shift:]  # B1_SW is cen's NE
    B1_SE = cen[:, :, :shift, :shift]  # B1_SE is cen's NW
    B1_N = torch.cat((B1_NW, B1_NE), dim=3)
    B1_S = torch.cat((B1_SW, B1_SE), dim=3)
    B1 = torch.cat((B1_N, B1_S), dim=2)

    ######## B2 #########
    # old: A  =>  new: B
    #      B  =>       A
    B2_N = cen[:, :, shift:, :]  # B2_N is cen's S
    B2_S = cen[:, :, :shift, :]  # B2_S is cen's N
    B2 = torch.cat((B2_N, B2_S), dim=2)

    ######## B3 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B3_NW = cen[:, :, shift:, wid - shift:]  # B3_NW is cen's SE
    B3_NE = cen[:, :, shift:, :wid - shift]  # B3_NE is cen's SW
    B3_SW = cen[:, :, :shift, wid - shift:]  # B3_SW is cen's NE
    B3_SE = cen[:, :, :shift, :wid - shift]  # B1_SE is cen's NW
    B3_N = torch.cat((B3_NW, B3_NE), dim=3)
    B3_S = torch.cat((B3_SW, B3_SE), dim=3)
    B3 = torch.cat((B3_N, B3_S), dim=2)

    ######## B4 #########
    # old: AB  =>  new: BA
    B4_W = cen[:, :, :, wid - shift:]  # B2_W is cen's E
    B4_E = cen[:, :, :, :wid - shift]  # B2_E is cen's S
    B4 = torch.cat((B4_W, B4_E), dim=3)

    ######## B5 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B5_NW = cen[:, :, hei - shift:, wid - shift:]  # B5_NW is cen's SE
    B5_NE = cen[:, :, hei - shift:, :wid - shift]  # B5_NE is cen's SW
    B5_SW = cen[:, :, :hei - shift, wid - shift:]  # B5_SW is cen's NE
    B5_SE = cen[:, :, :hei - shift, :wid - shift]  # B5_SE is cen's NW
    B5_N = torch.cat((B5_NW, B5_NE), dim=3)
    B5_S = torch.cat((B5_SW, B5_SE), dim=3)
    B5 = torch.cat((B5_N, B5_S), dim=2)

    ######## B6 #########
    # old: A  =>  new: B
    #      B  =>       A
    B6_N = cen[:, :, hei - shift:, :]  # B6_N is cen's S
    B6_S = cen[:, :, :hei - shift, :]  # B6_S is cen's N
    B6 = torch.cat((B6_N, B6_S), dim=2)

    ######## B7 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B7_NW = cen[:, :, hei - shift:, shift:]  # B7_NW is cen's SE
    B7_NE = cen[:, :, hei - shift:, :shift]  # B7_NE is cen's SW
    B7_SW = cen[:, :, :hei - shift, shift:]  # B7_SW is cen's NE
    B7_SE = cen[:, :, :hei - shift, :shift]  # B7_SE is cen's NW
    B7_N = torch.cat((B7_NW, B7_NE), dim=3)
    B7_S = torch.cat((B7_SW, B7_SE), dim=3)
    B7 = torch.cat((B7_N, B7_S), dim=2)

    ######## B8 #########
    # old: AB  =>  new: BA
    B8_W = cen[:, :, :, shift:]  # B8_W is cen's E
    B8_E = cen[:, :, :, :shift]  # B8_E is cen's S
    B8 = torch.cat((B8_W, B8_E), dim=3)

    return B1, B2, B3, B4, B5, B6, B7, B8


def cal_pcm(cen, shift):
    B1, B2, B3, B4, B5, B6, B7, B8 = circ_shift(cen, shift=shift)
    s1 = (B1 - cen) * (B5 - cen)
    s2 = (B2 - cen) * (B6 - cen)
    s3 = (B3 - cen) * (B7 - cen)
    s4 = (B4 - cen) * (B8 - cen)

    c12 = torch.minimum(s1, s2)
    c123 = torch.minimum(c12, s3)
    c1234 = torch.minimum(c123, s4)

    return c1234
