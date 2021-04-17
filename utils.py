import numpy as np
import random
import time

import torch as th

from pyevtk.hl import imageToVTK

'''
--- Single scalar field ---
'''

def field_and_grad_from_net(dataset, net, is_cuda, tiled_res=32):
    target_res = dataset.vol_res
    full_vol = th.zeros(target_res)
    target_res_pos = list(dataset.vol_res)
    target_res_pos.append(net.d_in)
    full_grad = th.zeros(target_res_pos)
    for xdx in np.arange(0,target_res[0],tiled_res):
        x_begin = xdx
        x_end = xdx+tiled_res if xdx+tiled_res <= target_res[0] else target_res[0]
        for ydx in np.arange(0,target_res[1],tiled_res):
            y_begin = ydx
            y_end = ydx+tiled_res if ydx+tiled_res <= target_res[1] else target_res[1]
            for zdx in np.arange(0,target_res[2],tiled_res):
                z_begin = zdx
                z_end = zdx+tiled_res if zdx+tiled_res <= target_res[2] else target_res[2]

                tile_resolution = th.tensor([x_end-x_begin,y_end-y_begin,z_end-z_begin],dtype=th.int)

                min_alpha_bb = th.tensor([x_begin/(target_res[0]-1),y_begin/(target_res[1]-1),z_begin/(target_res[2]-1)],dtype=th.float)
                max_alpha_bb = th.tensor([(x_end-1)/(target_res[0]-1),(y_end-1)/(target_res[1]-1),(z_end-1)/(target_res[2]-1)],dtype=th.float)
                min_bounds = dataset.min_bb + min_alpha_bb*(dataset.max_bb-dataset.min_bb)
                max_bounds = dataset.min_bb + max_alpha_bb*(dataset.max_bb-dataset.min_bb)

                tile_positions = dataset.scales.view(1,1,1,3)*dataset.tile_sampling(min_bounds,max_bounds,tile_resolution)
                if is_cuda:
                    tile_positions = tile_positions.unsqueeze(0).cuda()
                tile_positions.requires_grad = True
                tile_vol = net(tile_positions.unsqueeze(0)).squeeze(0).squeeze(-1)

                ones = th.ones_like(tile_vol)
                vol_grad = th.autograd.grad(outputs=tile_vol, inputs=tile_positions, grad_outputs=ones, retain_graph=False, create_graph=True, allow_unused=False)[0]
                full_grad[x_begin:x_end,y_begin:y_end,z_begin:z_end] = vol_grad.detach().cpu()

                full_vol[x_begin:x_end,y_begin:y_end,z_begin:z_end] = tile_vol.detach().cpu()
            #
        #
    #
    return full_vol,full_grad
#

def field_from_net(dataset, net, is_cuda, tiled_res=32, verbose=False):
    target_res = dataset.vol_res
    full_vol = th.zeros(target_res)
    for xdx in np.arange(0,target_res[0],tiled_res):
        if verbose:
            print('x',xdx,'/',target_res[0])
        x_begin = xdx
        x_end = xdx+tiled_res if xdx+tiled_res <= target_res[0] else target_res[0]
        for ydx in np.arange(0,target_res[1],tiled_res):
            y_begin = ydx
            y_end = ydx+tiled_res if ydx+tiled_res <= target_res[1] else target_res[1]
            for zdx in np.arange(0,target_res[2],tiled_res):
                z_begin = zdx
                z_end = zdx+tiled_res if zdx+tiled_res <= target_res[2] else target_res[2]

                tile_resolution = th.tensor([x_end-x_begin,y_end-y_begin,z_end-z_begin],dtype=th.int)

                min_alpha_bb = th.tensor([x_begin/(target_res[0]-1),y_begin/(target_res[1]-1),z_begin/(target_res[2]-1)],dtype=th.float)
                max_alpha_bb = th.tensor([(x_end-1)/(target_res[0]-1),(y_end-1)/(target_res[1]-1),(z_end-1)/(target_res[2]-1)],dtype=th.float)
                min_bounds = dataset.min_bb + min_alpha_bb*(dataset.max_bb-dataset.min_bb)
                max_bounds = dataset.min_bb + max_alpha_bb*(dataset.max_bb-dataset.min_bb)

                with th.no_grad():
                    tile_positions = dataset.scales.view(1,1,1,3)*dataset.tile_sampling(min_bounds,max_bounds,tile_resolution)
                    if is_cuda:
                        tile_positions = tile_positions.unsqueeze(0).cuda()
                    tile_vol = net(tile_positions.unsqueeze(0)).squeeze(0).squeeze(-1)
                    full_vol[x_begin:x_end,y_begin:y_end,z_begin:z_end] = tile_vol.cpu()
                #
            #
        #
    #
    return full_vol
#

def tiled_net_out(dataset, net, is_cuda, gt_vol=None, evaluate=True, write_vols=False, filename='vol'):
    net.eval()
    full_vol = field_from_net(dataset, net, is_cuda, tiled_res=32)
    psnr = 0
    print('writing to VTK...')
    if evaluate and gt_vol is not None:
        diff_vol = gt_vol - full_vol
        sqd_max_diff = (th.max(gt_vol)-th.min(gt_vol))**2
        #full_vol = full_vol.cpu().transpose(1,2).transpose(0,1).transpose(1,2)
        l1_diff = th.mean(th.abs(diff_vol))
        mse = th.mean(th.pow(diff_vol,2))
        psnr = 10*th.log10(sqd_max_diff/th.mean(diff_vol**2))
        print('PSNR:',psnr,'l1:',l1_diff,'mse:',mse,'rmse:',th.sqrt(mse))

    if write_vols:
        imageToVTK(filename, pointData = {'sf':full_vol.numpy()})
        if gt_vol is not None:
            imageToVTK('gt', pointData = {'sf':gt_vol.numpy()})
    #

    print('back to training...')
    net.train()
    return psnr
#
