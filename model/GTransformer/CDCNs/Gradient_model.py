import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        #The hyper parameters settting
        self.convs_list=nn.ModuleList()
        delta1=np.array([[[-1, 0, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, -1, 0], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, -1], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels,1)
        self.shifts=[1,5]
        self.scale=torch.nn.Parameter(torch.zeros(len(self.shifts)))
        #The Process the of extraction of outcome
        self.kernel1 = torch.Tensor(w1).cuda()
        self.kernel2 = torch.Tensor(w2).cuda()
        self.kernel3 = torch.Tensor(w3).cuda()
        self.kernel4 = torch.Tensor(w4).cuda()
        self.kernel5 = torch.Tensor(w5).cuda()
        self.kernel6 = torch.Tensor(w6).cuda()
        self.kernel7 = torch.Tensor(w7).cuda()
        self.kernel8 = torch.Tensor(w8).cuda()
        self.kernel1 = self.kernel1.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel2 = self.kernel2.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel3 = self.kernel3.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel4 = self.kernel4.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel5 = self.kernel5.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel6 = self.kernel6.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel7 = self.kernel7.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel8 = self.kernel8.repeat(self.in_channels, 1, 1, 1).contiguous()
        #After Extraction, we analyze the outcome of the extraction.
        self.num_layer=9
        self.act=nn.Softmax(dim=2)
        self.input_layers=nn.ModuleList()
        self.layers_1=nn.ModuleList()
        self.layers_2=nn.ModuleList()
        self.layers_3=nn.ModuleList()
        for shift in self.shifts:
            kernel=max(1,shift-2)
            self.input_layers.append(nn.Conv2d(in_channels=in_channels,out_channels=self.in_channels,kernel_size=kernel,stride=1,padding='same'))
            self.layers_1.append(nn.Conv2d(in_channels=self.in_channels*self.num_layer,out_channels=self.in_channels*self.num_layer,kernel_size=1,stride=1,groups=self.num_layer,bias=False))
            self.layers_2.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False))
            self.layers_3.append(nn.Conv2d(in_channels=self.in_channels*self.num_layer,out_channels=self.in_channels*self.num_layer,kernel_size=1,stride=1,groups=self.num_layer,bias=False))
    def initialize_biases(self, prior_prob):
        b = self.out_conv.bias.view(1, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.out_conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def Extract_layer(self,cen,i):
        surround1 = torch.nn.functional.conv2d(weight=self.kernel1, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
        surround2 = torch.nn.functional.conv2d(weight=self.kernel2, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
        surround3 = torch.nn.functional.conv2d(weight=self.kernel3, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
        surround4 = torch.nn.functional.conv2d(weight=self.kernel4, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
        surround5 = torch.nn.functional.conv2d(weight=self.kernel5, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
        surround6 = torch.nn.functional.conv2d(weight=self.kernel6, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
        surround7 = torch.nn.functional.conv2d(weight=self.kernel7, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
        surround8 = torch.nn.functional.conv2d(weight=self.kernel8, stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
        surrounds = torch.concat([surround1,surround2,surround3,surround4,surround5,surround6,surround7,surround8,cen],1)
        return surrounds
    def Analyze_layer(self,surrounds,center,i=0):
        b,c,w,h=surrounds.shape
        out1=self.layers_1[i](surrounds).view(b,self.num_layer,c//self.num_layer,w,h).contiguous()
        out2=self.layers_2[i](center).view(b,1,c//self.num_layer,w,h).contiguous()
        out3=self.layers_3[i](surrounds).view(b,self.num_layer,c//self.num_layer,w,h).contiguous()
        out1 = torch.nn.functional.normalize(out1,dim=2)
        out2 = torch.nn.functional.normalize(out2,dim=2)
        out3 = torch.nn.functional.normalize(out3,dim=2)
        attention=self.act(torch.mean(out1*out2,dim=2,keepdim=True))
        out=torch.sum(attention*out3,1).view(b,-1,w,h).contiguous()
        return out
    def forward(self,cen):
        # for i,shift in enumerate(self.shifts):
        surrounds = self.Extract_layer(cen=cen,i=0)
        outs = self.Analyze_layer(surrounds=surrounds,center=cen)+cen
        return outs