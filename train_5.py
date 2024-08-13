import argparse
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ISTD.net import Net
from utils.utils import seed_pytorch, get_optimizer
import numpy as np
import os

from utils.datasets import NUDTSIRSTSetLoader
from utils.datasets import IRSTD1KSetLoader
from utils.datasets import SIRSTAugSetLoader
from utils.datasets import SIRSTSetLoader
# from utils.datasets import NUAASIRSTSetLoader

from evaluation.mIoU import mIoU
from evaluation.pd_fa import PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP

torch.cuda.set_device(0)

# import warnings
# warnings.filterwarnings("ignore",category=UserWarning )

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch ISTD")

parser.add_argument("--model_names", default='ALCNet', type=str, nargs='+',
                    help="model_name: 'ALCNet', 'ACM', "
                         "'DNANet', 'AGPCNet'")
parser.add_argument("--dataset_names", default='NUDT-SIRST', type=str, nargs='+',
                    help="dataset_name: 'NUDT-SIRST', 'IRSTD-1K', 'SIRST-aug','SIRST','NUAA-SIRST'")

parser.add_argument("--dataset_dir", default='./data', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch sizse")
parser.add_argument("--save", default='./log5', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default=None, type=list, help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--nEpochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.1}, type=dict,
                    help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")

global opt
opt = parser.parse_args()
seed_pytorch(opt.seed)


def train():
    if opt.dataset_name == "NUDT-SIRST":
        dataset_dir = r'./data/NUDT-SIRST/'
        train_set = NUDTSIRSTSetLoader(base_dir=dataset_dir, mode='trainval')
    elif opt.dataset_name == "IRSTD-1K":
        dataset_dir = r'./data/IRSTD-1K/'
        train_set = IRSTD1KSetLoader(base_dir=dataset_dir, mode='trainval')
    elif opt.dataset_name == "SIRST-aug":
        dataset_dir = r'./data/sirst_aug/'
        train_set = SIRSTAugSetLoader(base_dir=dataset_dir, mode='trainval')
    elif opt.dataset_name == "SIRST":
        dataset_dir = r'./data/sirst/'
        train_set = SIRSTSetLoader(base_dir=dataset_dir, mode='trainval')
    # elif (opt.dataset_name == "NUAA-SIRST"):
    #     dataset_dir = r'./data/NUAA-SIRST/'
    #     train_set = NUAASIRSTSetLoader(base_dir=dataset_dir, mode='trainval')
    else:
        raise NotImplementedError

    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    net = Net(model_name=opt.model_name, mode='train').cuda()
    # print(net)
    net.train()

    epoch_state = 0
    opt.best_miou = 0
    opt.best_miou_epoch = 0
    opt.best_fscore = 0
    opt.best_fscore_epoch = 0

    opt.best_pd = 0
    opt.best_pd_epoch = 0
    opt.best_fa = 1
    opt.best_fa_epoch = 0

    total_loss_list = []
    total_loss_epoch = []

    if opt.resume:
        for resume_pth in opt.resume:
            if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])
                epoch_state = ckpt['epoch']
                total_loss_list = ckpt['total_loss']
                for i in range(len(opt.step)):
                    opt.step[i] = opt.step[i] - ckpt['epoch']

    ### Default settings
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs': 400, 'step': [200, 300], 'gamma': 0.1}
        # opt.scheduler_settings = {'epochs':100, 'step': [50, 75], 'gamma': 0.1}
        # opt.scheduler_settings = {'epochs':200, 'step': [100, 150], 'gamma': 0.1}

    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings['lr'] = 0.05
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings['epochs'] = 1500
        opt.scheduler_settings['min_lr'] = 1e-3

    opt.nEpochs = opt.scheduler_settings['epochs']

    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)

    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()
            if img.shape[0] == 1:
                continue
            pred = net.forward(img)
            loss = net.loss(pred, gt_mask)
            total_loss_epoch.append(loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if (idx_epoch + 1) % 1 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print('*********' + time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,*********'
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write('*********' + time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,,*********\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []

            save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '/' + str(idx_epoch + 1) + '.pth.tar'

            test_with_save(save_pth, idx_epoch, total_loss_list, net.state_dict())


def test_with_save(save_pth, idx_epoch, total_loss_list, net_state_dict):
    if opt.dataset_name == "NUDT-SIRST":
        dataset_dir = r'./data/NUDT-SIRST/'
        test_set = NUDTSIRSTSetLoader(base_dir=dataset_dir, mode='test')
    elif opt.dataset_name == "IRSTD-1K":
        dataset_dir = r'./data/IRSTD-1K/'
        test_set = IRSTD1KSetLoader(base_dir=dataset_dir, mode='test')
    elif opt.dataset_name == "SIRST-aug":
        dataset_dir = r'./data/sirst_aug/'
        test_set = SIRSTAugSetLoader(base_dir=dataset_dir, mode='test')
    elif opt.dataset_name == "SIRST":
        dataset_dir = r'./data/sirst/'
        test_set = SIRSTSetLoader(base_dir=dataset_dir, mode='test')
    # elif (opt.dataset_name == "NUAA-SIRST"):
    #     dataset_dir = r'./data/NUAA-SIRST/'
    #     test_set = NUAASIRSTSetLoader(base_dir=dataset_dir, mode='test')
    else:
        raise NotImplementedError
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    # ckpt = torch.load(save_pth)
    net.load_state_dict(net_state_dict)
    net.eval()


    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        with torch.no_grad():
            img = Variable(img).cuda()
            # pred = net(img)
            pred = net.forward(img)
            pred = pred[:, :, :size[0], :size[1]]

        gt_mask = gt_mask[:, :, :size[0], :size[1]]

        eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update(pred[0, 0, :, :].cpu().detach().numpy(), gt_mask[0, 0, :, :].detach().numpy(), size)
        # eval_my_PD_FA.update(pred[0, 0, :, :].cpu().detach().numpy(), gt_mask[0, 0, :, :].detach().numpy())
        eval_mIoU_P_R_F.update(labels=gt_mask[0, 0, :, :].detach().numpy(),
                               preds=pred[0, 0, :, :].cpu().detach().numpy())

    Yin_pixAcc, Yin_mIoU = eval_mIoU.get()
    pd, fa = eval_PD_FA.get()
    _, _, _, fscore = eval_mIoU_P_R_F.get()

    save_checkpoint({
        'epoch': idx_epoch + 1,
        'state_dict': net.state_dict(),
        'total_loss': total_loss_list,
    }, save_pth)

    if Yin_mIoU > opt.best_miou:
        opt.best_miou = Yin_mIoU
        opt.best_miou_epoch = idx_epoch + 1

    if fscore > opt.best_fscore:
        opt.best_fscore = fscore
        opt.best_fscore_epoch = idx_epoch + 1


    if pd > opt.best_pd:
        opt.best_pd = pd
        opt.best_pd_epoch = idx_epoch + 1

    if fa < opt.best_fa:
        opt.best_fa = fa
        opt.best_fa_epoch = idx_epoch + 1

    print('pixAcc %.6f, mIoU: %.6f' % (Yin_pixAcc, Yin_mIoU))
    opt.f.write('pixAcc %.6f, mIoU: %.6f' % (Yin_pixAcc, Yin_mIoU) + '\n')
    print('Pd: %.6f, Fa: %.8f, fscore: %.6f' % (pd, fa, fscore))
    opt.f.write('Pd: %.6f, Fa: %.8f, fscore: %.6f' % (pd, fa, fscore) + '\n')

    print('Best mIoU: %.6f,when Epoch=%d, Best fscore: %.6f,when Epoch=%d' % (opt.best_miou, opt.best_miou_epoch, opt.best_fscore, opt.best_fscore_epoch))
    opt.f.write('Best mIoU: %.6f,when Epoch=%d, Best fscore: %.6f,when Epoch=%d' % (opt.best_miou, opt.best_miou_epoch, opt.best_fscore, opt.best_fscore_epoch) + '\n')

    print('Best Pd: %.6f,when Epoch=%d, Best Fa: %.8f,when Epoch=%d' % (opt.best_pd, opt.best_pd_epoch, opt.best_fa, opt.best_fa_epoch))
    opt.f.write('Best Pd: %.6f,when Epoch=%d, Best Fa: %.8f,when Epoch=%d' % (opt.best_pd, opt.best_pd_epoch, opt.best_fa, opt.best_fa_epoch) + '\n')


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


# 训练多个模型
if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' +
                         (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()

# 命令行如下：
# 单个模型时：
# python train_5.py --model_names DNANet --dataset_names NUDT-SIRST
# 多个模型时：
# python train_5.py --model_names ACM ALCNet AGPCNet DNANet UIUNet --dataset_names NUDT-SIRST IRSTD-1K SIRST-aug




