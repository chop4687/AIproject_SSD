from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from torch.nn.parallel.data_parallel import DataParallel
from cocomini import COCODetection
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

coco = {
    'num_classes': 4,
    'max_iter': 50000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

dataset = 'VOC'
dataset_root = VOC_ROOT
basenet = 'vgg16_reducedfc.pth'
batch_size = 32
resume = None
start_iter = 0
num_workers = 4
cuda = True
lr = 1e-4
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1
save_folder = 'weights/'

if torch.cuda.is_available():
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(save_folder):
    os.mkdir(save_folder)


from torch import optim
def train():
    batch_size = 16
    start_iter = 0
    num_workers = 4
    cuda = True
    lr = 1e-4
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.1

    dataset = COCODetection(root='/home/junkyu/SSD2/coco2',transform=SSDAugmentation())

    cfg = voc
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    ssd_net.load_state_dict(torch.load('/home/junkyu/SSD2/weights/ssd300_mAP_77.43_v2.pth'))

    cfg = coco
    new_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

    new_net.vgg = ssd_net.vgg
    new_net.extras = ssd_net.extras
    new_net.loc = ssd_net.loc
    net = new_net
    #net.load_state_dict(torch.load('/home/junkyu/SSD2/weights/ssd300_4500.pth'))
    #for param in net.parameters():
    #    param.requires_grad = False

    # cfg = coco

    net.conf[0] = nn.Conv2d(512, cfg['num_classes'] * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    net.conf[1] = nn.Conv2d(1024, cfg['num_classes'] * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    net.conf[2] = nn.Conv2d(512, cfg['num_classes'] * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    net.conf[3] = nn.Conv2d(256, cfg['num_classes'] * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    net.conf[4] = nn.Conv2d(256, cfg['num_classes'] * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    net.conf[5] = nn.Conv2d(256, cfg['num_classes'] * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    if cuda:
        net = torch.nn.DataParallel(new_net)
        cudnn.benchmark = True

    if cuda:
        net = net.cuda()

    print('Initializing weights...')


    optimizer = optim.SGD(net.parameters(), lr=lr, momentum = momentum,
                          weight_decay=weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // batch_size

    step_index = 0

    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 5000, gamma=0.1)
    data_loader = data.DataLoader(dataset, batch_size,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(start_iter, cfg['max_iter']):
        if iteration != 0 and (iteration % epoch_size == 0):
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        if cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c

        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data
        conf_loss += loss_c.data

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')

        if iteration != 0 and iteration % 500 == 0:
            print('Saving state, iter:', iteration)
            torch.save(new_net.state_dict(), '/home/junkyu/SSD2/weights/ssdf300_' +
                       repr(iteration) + '.pth')
            scheduler.step()
    torch.save(new_net.state_dict(),
               save_folder + '' + dataset + '.pth')


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
