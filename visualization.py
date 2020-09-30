""""

Author: Xu Ma
Date: Aug/15/2019
Email: xuma@my.unt.edu

Useage:


"""


import argparse
import os
import shutil
import time
import math
import traceback
import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import models as models
from utils import Logger, mkdir_p, get_device, get_classtable,GradCAM


try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', default='/PATH_to_imageNet/ImageNet2012/', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names)
parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                    help='path to your checkpoint')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Running on GPU or CPU.')
parser.add_argument('-i', '--image', default='cat.png', type=str, metavar='PATH',
                    help='path to your image')
parser.add_argument('-o', '--output-dir', default='', type=str, metavar='PATH',
                    help='folder for save images')
parser.add_argument('-t', '--target-layer', default='layer4', type=str,
                    help='Target layer for visualization')



args = parser.parse_args()


def main():
    device = get_device(args.cuda)
    # Synset words
    classes = get_classtable()
    # Model from torchvision
    model = models.__dict__[args.arch]()
    model = model.cuda()
    check_point = torch.load(args.checkpoint, map_location=lambda storage, loc: storage.cuda(0))
    model.load_state_dict(check_point['state_dict'])
    model.to(device)
    model.eval()

    gcam = GradCAM(model=model)
    _ = gcam.forward(images)


if __name__ == '__main__':
    main()
