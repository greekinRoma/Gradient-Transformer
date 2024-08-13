import argparse
from ISTD.net import Net
import os
import time
from thop import profile
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model_name = 'ACM'
input_img = torch.rand(1,1,256,256).cuda()
net = Net(model_name, mode='test').cuda()
flops, params = profile(net, inputs=(input_img, ))
print(model_name)
print('Params: %2fM' % (params/1e6))
print('FLOPs: %2fGFLOPs' % (flops/1e9))

net.eval()
output = net(input_img)
# 获取输出图像大小
output_size = output.size()
print('Output Image Size:', output_size)