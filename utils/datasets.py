import os.path as osp
from utils.images import load_image
import matplotlib.pyplot as plt
import os

import cv2
import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


## Dataloader for deep learning
class MDFASetLoader(Dataset):
    def __init__(self, base_dir='../data/MDFA/', mode='test'):
        super(MDFASetLoader).__init__()
        assert mode in ['trainval', 'test']
        self.mode = mode
        if mode == 'trainval':
            self.img_dir = osp.join(base_dir, 'training')
            self.mask_dir = osp.join(base_dir, 'training')
            self.length = 9978
        elif mode == 'test':
            self.img_dir = osp.join(base_dir, 'test_org')
            self.mask_dir = osp.join(base_dir, 'test_gt')
            self.length = 100
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        if self.mode == 'trainval':
            img_path = osp.join(self.img_dir, '%06d_1.png' % i)
            mask_path = osp.join(self.mask_dir, '%06d_2.png' % i)
        elif self.mode == 'test':
            img_path = osp.join(self.img_dir, '%05d.png' % i)
            mask_path = osp.join(self.mask_dir, '%05d.png' % i)
        else:
            raise NotImplementedError

        img = load_image(img_path)
        mask = load_image(mask_path)
        img = np.array(img, dtype=np.float32)  / 255.0
        mask = np.array(mask, dtype=np.float32)  / 255.0

        if self.mode == 'trainval':
            h = 256
            w = 256
            img = cv2.resize(img, dsize=(h, w), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(h, w), interpolation = cv2.INTER_NEAREST)
        else:
            h, w = img.shape

        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h,w], self.names[i]

    def __len__(self):
        return self.length

class NUDTSIRSTSetLoader(Dataset):
    def __init__(self, base_dir='../data/NUDT-SIRST/', mode='test'):
        super(NUDTSIRSTSetLoader).__init__()
        self.mode = mode

        if mode == 'trainval':
            txtfile = 'train_NUDT-SIRST.txt'
        elif mode == 'test':
            txtfile = 'test_NUDT-SIRST.txt'
        else:
            raise NotImplementedError
        
        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32)  / 255.0
        mask = np.array(mask, dtype=np.float32)  / 255.0

        #h, w = img.shape # All the pictures are 256 x 256
        if self.mode == 'trainval':
            h = 256
            w = 256
            img = cv2.resize(img, dsize=(h, w), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(h, w), interpolation = cv2.INTER_NEAREST)
        else:
            h, w = img.shape

        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h,w], self.names[i]

    def __len__(self):
        return len(self.names)
    

class IRSTD1KSetLoader(Dataset):
    def __init__(self, base_dir='../data/IRSTD-1K/', mode='test'):
        super(IRSTD1KSetLoader).__init__()
        self.mode = mode

        if mode == 'trainval':
            txtfile = 'train_IRSTD-1K.txt'
        elif mode == 'test':
            txtfile = 'test_IRSTD-1K.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32)  / 255.0
        mask = np.array(mask, dtype=np.float32)  / 255.0

        h, w = img.shape # All the pictures are 512 x 512
        # if self.mode == 'trainval':
        #     h = 512
        #     w = 512
        #     img = cv2.resize(img, dsize=(h, w), interpolation = cv2.INTER_LINEAR)
        #     mask = cv2.resize(mask, dsize=(h, w), interpolation = cv2.INTER_NEAREST)
        # else:
        #     h, w = img.shape

        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h,w], self.names[i]

    def __len__(self):
        return len(self.names)
    
class SIRSTAugSetLoader(Dataset):
    def __init__(self, base_dir='../data/sirst_aug/', mode='test'):
        super(SIRSTAugSetLoader).__init__()
        self.mode = mode

        if mode == 'trainval':
            txtfile = 'train.txt'
        elif mode == 'test':
            txtfile = 'test.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        label_path = osp.join(self.label_dir, name + '_mask.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32)  / 255.0
        mask = np.array(mask, dtype=np.float32)  / 255.0

        #h, w = img.shape # All the pictures are 512 x 512
        if self.mode == 'trainval':
            h = 256
            w = 256
            img = cv2.resize(img, dsize=(h, w), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(h, w), interpolation = cv2.INTER_NEAREST)
        else:
            h, w = img.shape

        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h,w], self.names[i]

    def __len__(self):
        return len(self.names)

class SIRSTSetLoader(Dataset):
    def __init__(self, base_dir='../data/sirst/', mode='test'):
        super(SIRSTSetLoader).__init__()
        self.mode = mode
        if mode == 'trainval':
            txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'idx_427', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'_pixels0.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0


        if self.mode == 'trainval':
            h = 256
            w = 256
            img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
        else:
            h, w = img.shape

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h, w], self.names[i]

    def __len__(self):
        return len(self.names)

class NUAASIRSTSetLoader(Dataset):
    def __init__(self, base_dir='../data/NUAA-SIRST/', mode='test'):
        super(NUAASIRSTSetLoader).__init__()
        self.mode = mode
        if mode == 'trainval':
            txtfile = 'train_NUAA-SIRST.txt'
            # txtfile = 'trainval.txt'
        elif mode == 'test':
            txtfile = 'test_NUAA-SIRST.txt'
            # txtfile = 'test.txt'
        else:
            raise NotImplementedError

        self.list_dir = osp.join(base_dir, 'img_idx', txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')

        img = load_image(img_path)
        mask = load_image(label_path)
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0


        if self.mode == 'trainval':
            h = 256
            w = 256
            img = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=(h, w), interpolation=cv2.INTER_NEAREST)
        else:
            h, w = img.shape

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.mode == 'trainval':
            return img, mask
        else:
            return img, mask, [h, w], self.names[i]

    def __len__(self):
        return len(self.names)