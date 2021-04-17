import sys
import argparse
import os
import numpy as np
import time
import json
import re
import math
import struct

from sklearn.cluster import KMeans

import torch as th
import torch.nn as nn
import torch.backends.cudnn as cudnn

from siren import FieldNet

def get_weight_mats(net):
    weight_mats = [(name,parameters.data) for name, parameters in net.named_parameters() if re.match(r'.*.weight', name, re.I)]
    return [mat[1].cpu() for mat in weight_mats]
#

def get_bias_vecs(net):
    bias_vecs = [(name,parameters.data) for name, parameters in net.named_parameters() if re.match(r'.*.bias', name, re.I)]
    return [bias[1].cpu() for bias in bias_vecs]
#

def kmeans_quantization(w,q):
    weight_feat = w.view(-1).unsqueeze(1).numpy()
    kmeans = KMeans(n_clusters=q,n_init=4).fit(weight_feat)

    return kmeans.labels_.tolist(),kmeans.cluster_centers_.reshape(q).tolist()
#

def ints_to_bits_to_bytes(all_ints,n_bits):
    f_str = '#0'+str(n_bits+2)+'b'
    bit_string = ''.join([format(v, f_str)[2:] for v in all_ints])
    n_bytes = len(bit_string)//8
    the_leftover = len(bit_string)%8>0
    if the_leftover:
        n_bytes+=1
    the_bytes = bytearray()
    for b in range(n_bytes):
        bin_val = bit_string[8*b:] if b==(n_bytes-1) else bit_string[8*b:8*b+8]
        the_bytes.append(int(bin_val,2))
    #
    return the_bytes,the_leftover
#

class SimpleMap(dict):
    def __init__(self):
        pass

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(SimpleMap, self).__setitem__(key, value)
        self.__dict__.update({key: value})
#

class SirenEncoder:
    def __init__(self,net,config):
        self.net = net
        self.config = config
    #

    def encode(self,filename,n_bits,d_in=3):
        n_clusters = int(math.pow(2,n_bits))

        n_layers = self.config['n_layers']
        layers = self.config['layers']
        is_residual = 1 if self.config['is_residual'] else 0
        d_out = 1

        weight_mats = get_weight_mats(self.net)
        bias_vecs = get_bias_vecs(self.net)

        file = open(filename,'wb')

        # header: number of layers
        header = file.write(struct.pack('B', n_layers))
        # header: d_in
        header += file.write(struct.pack('B', d_in))
        # header: d_out
        header += file.write(struct.pack('B', d_out))
        # header: is_residual
        header += file.write(struct.pack('B', is_residual))
        # header: layers
        header += file.write(struct.pack(''.join(['I' for _ in range(len(layers))]), *layers))
        # header: number of bits for clustering
        header += file.write(struct.pack('B', n_bits))

        # first layer: matrix and bias
        w_pos,b_pos = weight_mats[0].view(-1).tolist(),bias_vecs[0].view(-1).tolist()
        w_pos_format = ''.join(['f' for _ in range(len(w_pos))])
        b_pos_format = ''.join(['f' for _ in range(len(b_pos))])
        first_layer = file.write(struct.pack(w_pos_format, *w_pos))
        first_layer += file.write(struct.pack(b_pos_format, *b_pos))

        # middle layers: cluster, store clusters, then map matrix indices to indices
        mid_bias,mid_weight=0,0
        for weight_mat,bias_vec in zip(weight_mats[1:-1],bias_vecs[1:-1]):
            labels,centers = kmeans_quantization(weight_mat,n_clusters)

            # weights
            w = centers
            w_format = ''.join(['f' for _ in range(len(w))])
            mid_weight += file.write(struct.pack(w_format, *w))
            weight_bin,is_leftover = ints_to_bits_to_bytes(labels,n_bits)
            mid_weight += file.write(weight_bin)

            # encode non-pow-2 as 16-bit integer
            if n_bits%8 != 0:
                mid_weight += file.write(struct.pack('I', labels[-1]))
            #

            # bias
            b = bias_vec.view(-1).tolist()
            b_format = ''.join(['f' for _ in range(len(b))])
            mid_bias += file.write(struct.pack(b_format, *b))
        #

        # last layer: matrix and bias
        w_last,b_last = weight_mats[-1].view(-1).tolist(),bias_vecs[-1].view(-1).tolist()
        w_last_format = ''.join(['f' for _ in range(len(w_last))])
        b_last_format = ''.join(['f' for _ in range(len(b_last))])
        last_layer = file.write(struct.pack(w_last_format, *w_last))
        last_layer += file.write(struct.pack(b_last_format, *b_last))

        file.flush()
        file.close()
    #
#

class SirenDecoder:
    def __init__(self):
        pass
    #

    def decode(self,filename):
        #weight_mats = get_weight_mats(self.net)
        #bias_vecs = get_bias_vecs(self.net)

        file = open(filename,'rb')

        # header: number of layers
        self.n_layers = struct.unpack('B', file.read(1))[0]
        # header: d_in
        self.d_in = struct.unpack('B', file.read(1))[0]
        # header: d_out
        self.d_out = struct.unpack('B', file.read(1))[0]
        # header: is_residual
        self.is_residual = struct.unpack('B', file.read(1))[0]
        # header: layers
        self.layers = struct.unpack(''.join(['I' for _ in range(self.n_layers)]), file.read(4*(self.n_layers)))
        # header: number of bits for clustering
        self.n_bits = struct.unpack('B', file.read(1))[0]
        self.n_clusters = int(math.pow(2,self.n_bits))
        print('n bits?',self.n_bits,'n clusters?',self.n_clusters)

        # create net from header
        opt = SimpleMap()
        self.d_in = 3
        opt.d_in = self.d_in
        opt.d_out = self.d_out
        opt.L = 0
        opt.w0 = 30
        opt.n_layers = self.n_layers
        opt.layers = self.layers
        opt.is_residual = self.is_residual==1

        net = FieldNet(opt)

        # first layer: matrix and bias
        w_pos_format = ''.join(['f' for _ in range(self.d_in*self.layers[0])])
        b_pos_format = ''.join(['f' for _ in range(self.layers[0])])
        w_pos = th.FloatTensor(struct.unpack(w_pos_format, file.read(4*self.d_in*self.layers[0])))
        b_pos = th.FloatTensor(struct.unpack(b_pos_format, file.read(4*self.layers[0])))

        all_ws = [w_pos]
        all_bs = [b_pos]

        # middle layers: cluster, store clusters, then map matrix indices to indices
        total_n_layers = 2*(self.n_layers-1) if self.is_residual==1 else self.n_layers-1
        for ldx in range(total_n_layers):
            # weights
            n_weights = self.layers[0]*self.layers[0]
            weight_size = (n_weights*self.n_bits)//8
            if (n_weights*self.n_bits)%8 != 0:
                weight_size+=1
            c_format = ''.join(['f' for _ in range(self.n_clusters)])
            centers = th.FloatTensor(struct.unpack(c_format, file.read(4*self.n_clusters)))
            inds = file.read(weight_size)
            bits = ''.join(format(byte, '0'+str(8)+'b') for byte in inds)
            w_inds = th.LongTensor([int(bits[self.n_bits*i:self.n_bits*i+self.n_bits],2) for i in range(n_weights)])

            if self.n_bits%8 != 0:
                next_bytes = file.read(4)
                w_inds[-1] = struct.unpack('I', next_bytes)[0]
            #

            # bias
            b_format = ''.join(['f' for _ in range(self.layers[0])])
            bias = th.FloatTensor(struct.unpack(b_format, file.read(4*self.layers[0])))

            w_quant = centers[w_inds]
            all_ws.append(w_quant)
            all_bs.append(bias)
        #

        # last layer: matrix and bias
        w_last_format = ''.join(['f' for _ in range(self.d_out*self.layers[-1])])
        b_last_format = ''.join(['f' for _ in range(self.d_out)])
        w_last = th.FloatTensor(struct.unpack(w_last_format, file.read(4*self.d_out*self.layers[-1])))
        b_last = th.FloatTensor(struct.unpack(b_last_format, file.read(4*self.layers[-1])))

        all_ws.append(w_last)
        all_bs.append(b_last)

        wdx,bdx=0,0
        for name, parameters in net.named_parameters():
            if re.match(r'.*.weight', name, re.I):
                w_shape = parameters.data.shape
                parameters.data = all_ws[wdx].view(w_shape)
                wdx+=1
            #
            if re.match(r'.*.bias', name, re.I):
                b_shape = parameters.data.shape
                parameters.data = all_bs[bdx].view(b_shape)
                bdx+=1
            #
        #

        return net
    #
#
