import torch.utils.data as Data
import torchvision.transforms as transforms

from utils.datasets import *
from PIL import Image, ImageOps, ImageFilter
import os
import os.path as osp
import scipy.io as scio
import numpy as np
import cv2

from evaluation.mIoU import mIoU
from evaluation.roc_cruve import ROCMetric
from evaluation.pd_fa import PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP
import argparse
import time

parser = argparse.ArgumentParser(description="PyTorch ISTD")
parser.add_argument("--model_names", default='ACM', type=str, nargs='+',
                    help="model_name: 'ALCNet', 'ACM', "
                         "'DNANet', 'AGPCNet'")
parser.add_argument("--dataset_names", default='IRSTD-1K', type=str, nargs='+',
                    help="dataset_name: 'NUDT-SIRST', 'IRSTD-1K', 'SIRST-aug','SIRST','NUAA-SIRST'")
parser.add_argument("--save", default='./result', type=str, help="Save path of results")
global opt
opt = parser.parse_args()


class Dataset_mat(Data.Dataset):
    def __init__(self, dataset, base_size=256, thre=0.5):

        self.base_size = base_size
        self.dataset = dataset
        if (dataset == 'NUDT-SIRST'):
            self.mat_dir = r'./result/NUDT-SIRST/mat/' + opt.model_name
            self.base_dir = r'./data/NUDT-SIRST/'
            txtfile = 'test_NUDT-SIRST.txt'
        elif (dataset == 'IRSTD-1K'):
            self.mat_dir = r'./result/IRSTD-1K/mat/' + opt.model_name
            self.base_dir = r'./data/IRSTD-1K/'
            txtfile = 'test_IRSTD-1K.txt'
            self.base_size = 512
        elif (dataset == 'SIRST-aug'):
            self.mat_dir = r'./result/SIRST-aug/mat/' + opt.model_name
            self.base_dir = r'./data/sirst_aug/'
            txtfile = 'test.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(self.base_dir, 'img_idx', txtfile)
        self.mask_dir = osp.join(self.base_dir, 'masks')

        file_mat_names = os.listdir(self.mat_dir)
        self.file_names = [s[:-4] for s in file_mat_names]

        self.thre = thre

        self.mat_transform = transforms.Resize((base_size, base_size), interpolation=Image.BILINEAR)
        self.mask_transform = transforms.Resize((base_size, base_size), interpolation=Image.NEAREST)

    def __getitem__(self, i):
        file_name = self.file_names[i]
        if (self.dataset == 'SIRST-aug'):
            mask_path = osp.join(self.mask_dir, file_name) + "_mask.png"
        else:
            mask_path = osp.join(self.mask_dir, file_name) + ".png"
        mat_path = osp.join(self.mat_dir, file_name) + ".mat"

        # print(mask_path)

        rstImg = scio.loadmat(mat_path)['T']
        rstImg = np.asarray(rstImg)

        rst_seg = np.zeros(rstImg.shape)
        rst_seg[rstImg > self.thre] = 1

        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), -1)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask / mask.max()

        rstImg = cv2.resize(rstImg, dsize=(self.base_size, self.base_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.base_size, self.base_size), interpolation=cv2.INTER_NEAREST)

        return rstImg, mask

    def __len__(self):
        return len(self.file_names)

def cal_fpr_tpr():

    # f = open(fileName, mode = 'a+')
    print('Running data: {:s}'.format(opt.dataset_name))
    opt.f.write('Running data: {:s}'.format(opt.dataset_name) + '\n')

    thre = 0.5

    #metrics = ROCMetric(nclass=1, bins=nbins)
    if (opt.dataset_name == 'IRSTD-1K'):
        baseSize =512
    else:
        baseSize =256

    dataset = Dataset_mat(opt.dataset_name, base_size=baseSize, thre=thre)

    roc = ROCMetric(bins=200)
    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

    for i in range(dataset.__len__()):
        rstImg, mask = dataset.__getitem__(i)
        size = rstImg.shape
        roc.update(pred=rstImg, label=mask)
        eval_mIoU.update((torch.from_numpy(rstImg.reshape(1,1,baseSize, baseSize))>thre), torch.from_numpy(mask.reshape(1,1,baseSize, baseSize)))
        eval_PD_FA.update(rstImg, mask, size)
        eval_mIoU_P_R_F.update(labels=mask, preds=rstImg)

    Yin_pixAcc, Yin_mIoU = eval_mIoU.get()
    fpr, tpr, auc = roc.get()
    pd, fa = eval_PD_FA.get()
    _, _, _, fscore = eval_mIoU_P_R_F.get()

    print('pixAcc %.6f, mIoU: %.6f, AUC: %.6f' % (Yin_pixAcc, Yin_mIoU, auc))
    opt.f.write('pixAcc %.6f, mIoU: %.6f, AUC: %.6f' % (Yin_pixAcc, Yin_mIoU, auc) + '\n')
    print('Pd: %.6f, Fa: %.8f, fscore: %.6f' % (pd, fa, fscore))
    opt.f.write('Pd: %.6f, Fa: %.8f, fscore: %.6f' % (pd, fa, fscore) + '\n')
    opt.f.write('\n')

    save_dict = {'tpr': tpr, 'fpr': fpr}
    scio.savemat(osp.join('./result', '{:s}_{:s}.mat'.format(opt.dataset_name,opt.model_name)), save_dict)


if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)

            opt.f = open(
                opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (
                    time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(
                opt.dataset_name + '\t' + opt.model_name)
            cal_fpr_tpr()
            print('\n')
            opt.f.close()

# python cal_metrics.py --model_names ACM ALCNet AGPCNet DNANet UIUNet --dataset_names NUDT-SIRST IRSTD-1K SIRST-aug