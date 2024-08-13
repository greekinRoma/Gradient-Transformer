import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F1

__all__ = ['NonLocalBlock', 'GCA_Channel', 'GCA_Element', 'AGCB_Element', 'AGCB_Patch', 'CPM', 'AGCB_Patch_my1', 'AGCB_Patch_my2', 'CPM_my']


class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x
        return out


class GCA_Channel(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Channel, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
        else:
            raise NotImplementedError
        return gca


class GCA_Element(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Element, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
            )
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        batch_size, C, height, width = x.size()

        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        else:
            raise NotImplementedError
        return gca


class AGCB_Patch(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Patch, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Channel(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block) * attention)

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class AGCB_Element(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Element, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Element(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            # attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block))

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = context * gca
        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class AGCB_NoGCA(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32):
        super(AGCB_NoGCA, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            context_list.append(self.non_local(block))

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class CPM(nn.Module):
    def __init__(self, planes, block_type, scales=(3,5,6,10), reduce_ratios=(4,8), att_mode='origin'):
        super(CPM, self).__init__()
        assert block_type in ['patch', 'element']
        assert att_mode in ['origin', 'post']

        inter_planes = planes // reduce_ratios[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, inter_planes, kernel_size=1),
            nn.BatchNorm2d(inter_planes),
            nn.ReLU(True),
        )

        if block_type == 'patch':
            self.scale_list = nn.ModuleList(
                [AGCB_Patch(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        elif block_type == 'element':
            self.scale_list = nn.ModuleList(
                [AGCB_Element(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        else:
            raise NotImplementedError

        channels = inter_planes * (len(scales) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        reduced = self.conv1(x)

        blocks = []
        for i in range(len(self.scale_list)):
            blocks.append(self.scale_list[i](reduced))
        out = torch.cat(blocks, 1)
        out = torch.cat((reduced, out), 1)
        out = self.conv2(out)
        return out

## mynet  ##############################################################################################################

from torch.nn import Softmax
def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, reduce_ratio=8):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class CrissCrossAttention_fake_x(nn.Module):
    def __init__(self, in_channels, reduce_ratio=8):
        super(CrissCrossAttention_fake_x, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(4*in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, H * W)
        proj_value = self.value_conv(x).view(B, -1, H * W)
        # proj_value = self.value_conv(x).view(B, H * W, H, W) # 可以看看怎么改。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。

        energy = torch.bmm(proj_query, proj_key)
        # energy = (self.INF(x) + energy).view(B, H * W, H, W)
        energy = energy.view(B, H * W, H, W)
        attention = F.softmax(energy, dim=-1)
        # #test
        # mm = attention.permute(0, 3, 2, 1)
        # mmm= mm.reshape(B, H * W, H * W)
        # hh = attention.permute(0, 2, 3, 1)
        # hhh= hh.view(B, H * W, H * W)

        # horizontal attention
        proj_h_value = torch.bmm(proj_value, attention.permute(0, 2, 3, 1).reshape(B, H * W, -1))
        proj_h_value = proj_h_value.view(B, -1, H, W)

        # vertical attention
        proj_v_value = torch.bmm(proj_value, attention.permute(0, 3, 2, 1).reshape(B, H * W, -1))
        proj_v_value = proj_v_value.view(B, -1, H, W)

        # diagonal attention
        proj_d_value = torch.bmm(proj_value, attention.permute(0, 2, 1, 3).reshape(B, H * W, -1))
        proj_d_value = proj_d_value.view(B, -1, H, W)

        # anti-diagonal attention
        proj_a_value = torch.bmm(proj_value, attention.permute(0, 3, 1, 2).reshape(B, H * W, -1))
        proj_a_value = proj_a_value.view(B, -1, H, W)

        out = torch.cat([proj_h_value, proj_v_value, proj_d_value, proj_a_value], dim=1)
        out = self.out_conv(out)
        out = self.gamma * out + x
        return out

class CrissCrossAttention_affine_x(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, reduce_ratio=8):
        super(CrissCrossAttention_affine_x,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        x_rotate = F1.rotate(x, -45)
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
########################################################################################################################
        proj_rotate_query = self.query_conv(x_rotate)
        proj_rotate_query_H = proj_rotate_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_rotate_query_W = proj_rotate_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_rotate_key = self.key_conv(x_rotate)
        proj_rotate_key_H = proj_rotate_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_rotate_key_W = proj_rotate_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_rotate_value = self.value_conv(x_rotate)
        proj_rotate_value_H = proj_rotate_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_rotate_value_W = proj_rotate_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_rotate_H = (torch.bmm(proj_rotate_query_H, proj_rotate_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_rotate_W = torch.bmm(proj_rotate_query_W, proj_rotate_key_W).view(m_batchsize, height, width, width)
        concate_rotate = self.softmax(torch.cat([energy_rotate_H, energy_rotate_W], 3))

        att_rotate_H = concate_rotate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_rotate_W = concate_rotate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_rotate_H = torch.bmm(proj_rotate_value_H, att_rotate_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_rotate_W = torch.bmm(proj_rotate_value_W, att_rotate_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        out1 = self.gamma1*(out_H + out_W)
        out2 = self.gamma2*(out_rotate_H + out_rotate_W)
        out2 = F1.rotate(out2, 45)
        return out1 + out2 + x


class AGCB_Patch_my1(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Patch_my1, self).__init__()

        self.scale = scale
        # self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        # self.non_local = CrissCrossAttention(planes, reduce_ratio=reduce_ratio_nl)
        # self.non_local = CrissCrossAttention_fake_x(planes, reduce_ratio=reduce_ratio_nl)
        self.non_local = CrissCrossAttention_affine_x(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Channel(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block) * attention)

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context

class AGCB_Patch_my2(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Patch_my2, self).__init__()

        self.scale = scale
        # self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        # self.non_local = CrissCrossAttention(planes, reduce_ratio=reduce_ratio_nl)
        # self.non_local = CrissCrossAttention_fake_x(planes, reduce_ratio=reduce_ratio_nl)
        self.non_local = CrissCrossAttention_affine_x(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Channel(planes, scale+1, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x) ############3 可能有问题，尺度不匹配

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_add_x, local_add_y, attention_add_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale

        step_h_add, step_w_add = step_h // 2, step_w // 2
        for i in range(self.scale+1):
            for j in range(self.scale+1):
                start_x, start_y = i * step_h, j * step_w

                if i == 0 :
                    start_add_x = 0
                    end_add_x = step_h_add
                elif i == (self.scale):
                    start_add_x = (i-1) * step_h + step_h_add
                    end_add_x = height
                else :
                    start_add_x = (i-1) * step_h + step_h_add
                    end_add_x = start_x + step_h_add

                if j == 0 :
                    start_add_y = 0
                    end_add_y = step_w_add
                elif j == (self.scale):
                    start_add_y = (j-1) * step_w + step_w_add
                    end_add_y = width
                else :
                    start_add_y = (j-1) * step_w + step_w_add
                    end_add_y = start_y + step_w_add

                local_add_x += [start_add_x, end_add_x]
                local_add_y += [start_add_y, end_add_y]

                attention_add_ind += [i, j]

        # index_cnt = 2 * self.scale * self.scale
        index_add_cnt = 2 * (self.scale + 1) * (self.scale + 1)
        # assert len(local_x) == index_cnt
        assert len(local_add_x) == index_add_cnt

        context_add_list = []
        for i in range(0, index_add_cnt, 2):
            block_add = x[:, :, local_add_x[i]:local_add_x[i+1], local_add_y[i]:local_add_y[i+1]]
            attention_add = gca[:, :, attention_add_ind[i], attention_add_ind[i+1]].view(batch_size, C, 1, 1)
            context_add_list.append(self.non_local(block_add) * attention_add)

        tmp_add = []
        for i in range(self.scale+1):
            row_add_tmp = []
            for j in range(self.scale+1):
                row_add_tmp.append(context_add_list[j + i * (self.scale+1)])
            tmp_add.append(torch.cat(row_add_tmp, 3))
        context = torch.cat(tmp_add, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context

class CPM_my(nn.Module):
    def __init__(self, planes, block_type, scales=(3,5,6,10), reduce_ratios=(4,8), att_mode='post'):
        super(CPM_my, self).__init__()
        assert block_type in ['patch', 'element']
        assert att_mode in ['origin', 'post']

        inter_planes = planes // reduce_ratios[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, inter_planes, kernel_size=1),
            nn.BatchNorm2d(inter_planes),
            nn.ReLU(True),
        )

        if block_type == 'patch':
            self.scale_list1 = nn.ModuleList(
                [AGCB_Patch_my1(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
            self.scale_list2 = nn.ModuleList(
                [AGCB_Patch_my2(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        # elif block_type == 'element':
        #     self.scale_list = nn.ModuleList(
        #         [AGCB_Element(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
        #          for scale in scales])
        else:
            raise NotImplementedError

        channels = inter_planes * (len(scales) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        reduced = self.conv1(x)

        blocks = []
        for i in range(len(self.scale_list1)):
            blocks_1 = self.scale_list1[i](reduced)
            blocks_2 = self.scale_list2[i](blocks_1)
            blocks.append(blocks_2)
        out = torch.cat(blocks, 1)
        out = torch.cat((reduced, out), 1)
        out = self.conv2(out)
        return out













