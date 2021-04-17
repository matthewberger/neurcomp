from __future__ import print_function
import argparse
import sys
import os
import numpy as np
import random
import time
import json
import re
import math

from sklearn.cluster import KMeans

import torch as th
from torch.utils.data import DataLoader

from utils import tiled_net_out

from data import VolumeDataset

from func_eval import trilinear_f_interpolation,finite_difference_trilinear_grad

from siren import FieldNet, compute_num_neurons
from net_coder import SirenDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--volume', required=True, help='path to volumetric dataset')
parser.add_argument('--compressed', required=True, help='path to compressed file')
parser.add_argument('--recon', default='recon', help='path to reconstructed file output')

parser.add_argument('--cuda', dest='cuda', action='store_true', help='enables cuda')
parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='disables cuda')
parser.set_defaults(cuda=False)

opt = parser.parse_args()
print(opt)

decoder = SirenDecoder()
net = decoder.decode(opt.compressed)
if opt.cuda:
    net = net.cuda()
net.eval()

# volume
np_volume = np.load(opt.volume).astype(np.float32)
volume = th.from_numpy(np_volume)
vol_res = th.prod(th.tensor([val for val in volume.shape])).item()

v_size = vol_res*4
compressed_size = os.path.getsize(opt.compressed)
cr = v_size/compressed_size
print('compression ratio:',cr)

raw_min = th.tensor([th.min(volume)],dtype=volume.dtype)
raw_max = th.tensor([th.max(volume)],dtype=volume.dtype)
volume = 2.0*((volume-raw_min)/(raw_max-raw_min)-0.5)

dataset = VolumeDataset(volume,16)

tiled_net_out(dataset, net, opt.cuda, gt_vol=volume, evaluate=True, write_vols=True, filename=opt.recon)
