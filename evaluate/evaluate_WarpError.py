#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib

import networks
import utils
import torch.nn as nn
import torch.nn.functional as F

class Resample2d(nn.Module):
    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2, kernel_size=None):
        if kernel_size is not None:
            self.kernel_size = kernel_size

        if input2.size(1) != 2:
            raise ValueError("input2 应该是光流张量，形状为 (B, 2, H, W)")

        B, _, H, W = input1.size()
        _, _, H_flow, W_flow = input2.size()
        device = input1.device

        if H != H_flow or W != W_flow:
            input2 = F.interpolate(input2, size=(H, W), mode='bilinear', align_corners=False)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=device),
            torch.arange(0, W, device=device),
            indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float()  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        flow = input2.permute(0, 2, 3, 1)  # (B, H, W, 2)

        norm_flow = torch.stack([
            2.0 * (flow[..., 0] / max(W - 1, 1)) - 1.0,
            2.0 * (flow[..., 1] / max(H - 1, 1)) - 1.0
        ], dim=-1)

        grid = grid + norm_flow

        warped_input = F.grid_sample(
            input1, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        return warped_input

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fast Blind Video Temporal Consistency')
    

    ### testing options
    parser.add_argument('-task',            type=str,     default="depth_estimation",            help='evaluated task')
    parser.add_argument('-method',          type=str,     default="marigold",            help='test model name')
    parser.add_argument('-dataset',         type=str,     default="ship",            help='test datasets')
    parser.add_argument('-phase',           type=str,     default="test",           choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to list folder')
    parser.add_argument('-redo',            action="store_true",                    help='redo evaluation')

    opts = parser.parse_args()
    opts.cuda = True

    print(opts)

    # output_dir = os.path.join(opts.data_dir, opts.phase, opts.method, opts.task, opts.dataset)
    # output_dir = os.path.join(opts.data_dir, opts.phase)
    #
    # if not os.path.isdir(output_dir):
    #     os.makedirs(output_dir)

    ## print average if result already exists
    metric_filename = os.path.join(opts.data_dir, opts.phase+'early', "WarpError.txt")
    if os.path.exists(metric_filename) and not opts.redo:
        print("Output %s exists, skip..." %metric_filename)

        cmd = 'tail -n1 %s' %metric_filename
        utils.run_cmd(cmd)
        sys.exit()
    

    ## flow warping layer
    device = torch.device("cuda" if opts.cuda else "cpu")
    flow_warping = Resample2d().to(device)

    ### load video list
    list_filename = os.path.join(opts.list_dir, "%s_%s.txt" %(opts.dataset, opts.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

    ### start evaluation
    err_all = np.zeros(len(video_list))

    for v in range(len(video_list)):

        video = video_list[v]

        frame_dir = os.path.join(opts.data_dir, opts.phase+'early', "processed\\depth_estimation\\", opts.dataset, video)
        # frame_dir = os.path.join(opts.data_dir, opts.phase, "outputatlas")

        occ_dir = os.path.join(opts.data_dir, opts.phase, "fw_occlusion", opts.dataset, video)
        flow_dir = os.path.join(opts.data_dir, opts.phase, "fw_flow", opts.dataset, video)
        
        frame_list = glob.glob(os.path.join(frame_dir, "*.png"))

        err = 0
        print(len(frame_list))
        for t in range(2, len(frame_list)+1):
            
            
            ### load input images
            filename = os.path.join(frame_dir, "%06d.png" %(t - 1))
            img1 = utils.read_img(filename)
            filename = os.path.join(frame_dir, "%06d.png" %(t))
            img2 = utils.read_img(filename)

            print("Evaluate Warping Error on %s-%s: video %d / %d, %s" %(opts.dataset, opts.phase, v + 1, len(video_list), filename))


            ### load flow
            filename = os.path.join(flow_dir, "%06d.flo" %(t-1))
            flow = utils.read_flo(filename)

            ### load occlusion mask
            filename = os.path.join(occ_dir, "%06d.png" %(t-1))
            occ_mask = utils.read_img(filename)
            noc_mask = 1 - occ_mask

            with torch.no_grad():

                ## convert to tensor
                img2 = utils.img2tensor(img2).to(device)
                flow = utils.img2tensor(flow).to(device)

                ## warp img2
                warp_img2 = flow_warping(img2, flow)

                ## convert to numpy array
                warp_img2 = utils.tensor2img(warp_img2)


            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            
            N = np.sum(noc_mask)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            err += np.sum(np.square(diff)) / N
            print(err)

        err_all[v] = err / (len(frame_list) - 1)


    print("\nAverage Warping Error = %f\n" %(err_all.mean()))

    err_all = np.append(err_all, err_all.mean())
    print("Save %s" %metric_filename)
    np.savetxt(metric_filename, err_all, fmt="%f")
