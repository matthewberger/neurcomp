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

import torch as th
from torch.utils.data import DataLoader

from siren import FieldNet, compute_num_neurons
from net_coder import SirenEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--net', required=True, help='path to trained network')
parser.add_argument('--config', required=True, help='path to network config')
parser.add_argument('--compressed', required=True, help='path to compressed file for output')
parser.add_argument('--cluster_bits', type=int, default=9, help='number of bits for cluster (2^b clusters)')

opt = parser.parse_args()
print(opt)

config = json.load(open(opt.config,'r'))

# config
opt.d_in = 3
opt.d_out = 1
opt.L = 0
opt.w0 = config['w0']
opt.n_layers = config['n_layers']
opt.layers = config['layers']
opt.compression_ratio = config['compression_ratio']
opt.oversample = config['oversample']
opt.cuda = config['is_cuda']
opt.is_residual = config['is_residual']

# network
net = FieldNet(opt)
net.load_state_dict(th.load(opt.net))
if opt.cuda:
    net = net.cuda()
net.eval()

encoder = SirenEncoder(net, config)
encoder.encode(opt.compressed,opt.cluster_bits)
