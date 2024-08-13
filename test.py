import argparse
import cv2
import torch
import scipy.io as scio
import os

from ISTD.net import Net
from utils.utils import seed_pytorch, get_optimizer
from utils.datasets import NUDTSIRSTSetLoader
from utils.datasets import IRSTD1KSetLoader
from utils.datasets import SIRSTAugSetLoader
from utils.datasets import SIRSTSetLoader
from torch.autograd import Variable
from torch.utils.data import DataLoader

# LS的参数
parser = argparse.ArgumentParser(description="PyTorch ISTD")

parser.add_argument("--model_names", default='ACM', type=str, nargs='+',
                    help="model_name: 'ALCNet', 'ACM', "
                         "'DNANet', 'AGPCNet'")
parser.add_argument("--dataset_names", default='NUDT-SIRST', type=str, nargs='+',
                    help="dataset_name: 'NUDT-SIRST', 'IRSTD-1K', 'SIRST-aug','SIRST','NUAA-SIRST'")
parser.add_argument("--dataset_dir", default='./data', type=str, help="train_dataset_dir")
parser.add_argument("--save", default='./log5', type=str, help="Save path of checkpoints")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--test_epo", type=str, default='30', help="Number of epoch for test")

global opt
opt = parser.parse_args()
seed_pytorch(opt.seed)



def test():
    if (opt.dataset_name == "NUDT-SIRST"):
        dataset_dir = r'./data/NUDT-SIRST/'
        test_set = NUDTSIRSTSetLoader(base_dir=dataset_dir, mode='test')
    elif (opt.dataset_name == "IRSTD-1K"):
        dataset_dir = r'./data/IRSTD-1K/'
        test_set = IRSTD1KSetLoader(base_dir=dataset_dir, mode='test')
    elif (opt.dataset_name == "SIRST-aug"):
        dataset_dir = r'./data/sirst_aug/'
        test_set = SIRSTAugSetLoader(base_dir=dataset_dir, mode='test')
    else:
        raise NotImplementedError

    param_path = "log5/" + opt.dataset_name + "/" + opt.model_name + '/' + opt.test_epo + '.pth.tar'

    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    # net = Net(model_name=opt.model_name, mode='test').cuda(device=0)
    net = Net(model_name=opt.model_name, mode='test').cuda(device=0)
    # ckpt = torch.load(save_pth)
    net.load_state_dict(torch.load(param_path, map_location='cuda:0')['state_dict'], False)
    net.eval()
    
    # print(opt.model_name)
    # print(opt.dataset_name)
    print('testing data=' + opt.dataset_name + ', model=' + opt.model_name + ', epoch=' + opt.test_epo)

    # 输出图片及图片对应mat文件的保存地址
    imgDir = "./result/" + opt.dataset_name + "/img/" + opt.model_name + "/"
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)
    matDir = "./result/" + opt.dataset_name + "/mat/" + opt.model_name + "/"
    if not os.path.exists(matDir):
        os.makedirs(matDir)
        
    for idx_iter, (img, gt_mask, size, iname) in enumerate(test_loader):
        name = iname[0]
        # name = os.path.splitext(iname)[0]
        # name = str(idx_iter)
        pngname = name + ".png"
        matname = name + '.mat'
        with torch.no_grad():
            img = Variable(img).cuda(device=0)
            # pred = net(img)
            pred = net.forward(img)
            pred = pred[:, :, :size[0], :size[1]]
            pred_out = pred.data.cpu().detach().numpy().squeeze()
            pred_out_png = pred_out * 255

        cv2.imwrite(imgDir + pngname, pred_out_png)
        scio.savemat(matDir + matname, {'T': pred_out})


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            test()
            print('\n')

# 其他网络
# python test.py --model_names ACM ALCNet AGPCNet DNANet UIUNet --dataset_names NUDT-SIRST IRSTD-1K SIRST-aug --test_epo 400