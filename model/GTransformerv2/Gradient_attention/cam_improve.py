import torch
from torch import nn
import numpy as np
import math
class GTransformer(nn.Module):
    def __init__(self, in_channels,out_channels,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        delta1=np.array([[[-1, 0, 0], [0, 0, 0], [0, 0, 0]], 
                         [[0, -1, 0], [0, 0, 0], [0, 0, 0]], 
                         [[0, 0, -1], [0, 0, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 0, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels//8,1)
        self.shifts=[1,5]
        self.scale=torch.nn.Parameter(torch.zeros(len(self.shifts)))
        self.hidden_channels=max(self.in_channels,1)
        #The Process the of extraction of outcome
            self.kernel1 = torch.Tensor(w1).cuda()
            self.kernel2 = torch.Tensor(w2).cuda()
            self.kernel3 = torch.Tensor(w3).cuda()
            self.kernel4 = torch.Tensor(w4).cuda()
            self.kernel5 = torch.Tensor(w5).cuda()
            self.kernel6 = torch.Tensor(w6).cuda()
            self.kernel7 = torch.Tensor(w7).cuda()
            self.kernel8 = torch.Tensor(w8).cuda()