import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
from torchvision import models

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = coco
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'val':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "val":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

def multibox(num_classes):
    from torchvision import models
    import torch.nn as nn
    vgg = []
    vgg_pretrained = models.vgg16(pretrained=True).features[:-1]
    vgg_pretrained[16] = nn.MaxPool2d(kernel_size=2, stride=2,padding=0, dilation=1 , ceil_mode=True)
    for i in range(len(vgg_pretrained)) :
        vgg += [vgg_pretrained[i]]
    vgg += [nn.MaxPool2d(kernel_size=3, stride=1,padding=1, dilation=1 , ceil_mode=False)]
    vgg += [nn.Conv2d(512,1024,kernel_size=(3,3),stride=(1,1),padding=(6,6),dilation=(6,6))]
    vgg += [nn.ReLU(inplace=True)]
    vgg += [nn.Conv2d(1024,1024,kernel_size=(1,1),stride=(1,1))]
    vgg += [nn.ReLU(inplace=True)]

    extra_layers = []
    extra_layers += [nn.Conv2d(1024, 256, kernel_size=(1, 1),stride=(1,1))]
    extra_layers += [nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2,2), padding=(1,1))]
    extra_layers += [nn.Conv2d(512, 128, kernel_size=(1, 1),stride=(1,1))]
    extra_layers += [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2,2), padding=(1,1))]
    extra_layers += [nn.Conv2d(256, 128, kernel_size=(1, 1),stride=(1,1))]
    extra_layers += [nn.Conv2d(128, 256, kernel_size=(3, 3),stride=(1,1))]
    extra_layers += [nn.Conv2d(256, 128, kernel_size=(1, 1),stride=(1,1))]
    extra_layers += [nn.Conv2d(128, 256, kernel_size=(3, 3),stride=(1,1))]


    loc_layers = []
    loc_layers += [nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    loc_layers += [nn.Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    loc_layers += [nn.Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    loc_layers += [nn.Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    loc_layers += [nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    loc_layers += [nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]

    conf_layers = []
    conf_layers += [nn.Conv2d(512, num_classes * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    conf_layers += [nn.Conv2d(1024, num_classes * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    conf_layers += [nn.Conv2d(512, num_classes * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    conf_layers += [nn.Conv2d(256, num_classes * 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    conf_layers += [nn.Conv2d(256, num_classes * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]
    conf_layers += [nn.Conv2d(256, num_classes * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))]

    return vgg, extra_layers, (loc_layers, conf_layers)


def build_ssd(phase, size=300, num_classes=21):
    base_, extras_, head_ = multibox(num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
