#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

### custom lib
import networks
import utils
from networks.src.stage_1.raft_wrapper import RAFTWrapper


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='optical flow estimation')

    ### testing options

    parser.add_argument('-dataset',         type=str,     default="ship",        help='testing datasets')
    parser.add_argument('-phase',           type=str,     default="test",       choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='data',       help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',      help='path to list folder')
    parser.add_argument('-gpu',             type=int,     default=0,            help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                help='use cpu?')
    parser.add_argument('-max_long_edge', type=int, default='2000',
                        help='maximum image dimension to process without resizing')

    opts = parser.parse_args()

    ### update options
    opts.cuda = (opts.cpu != True)
    opts.grads = {} # dict to collect activation gradients (for training debug purpose)

    opts.rgb_max = 1.0
    opts.fp16 = False

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")
    




    raft_wrapper = RAFTWrapper(
        model_path='../pretrained_weights/raft-things.pth', max_long_edge=opts.max_long_edge
    )
    model=raft_wrapper.compute_flow

    device = torch.device("cuda" if opts.cuda else "cpu")


    ### load image list
    list_filename = os.path.join(opts.list_dir, "%s_%s.txt" %(opts.dataset, opts.phase))
    with open(list_filename) as f:
        video_list = [line.rstrip() for line in f.readlines()]

   
    for video in video_list:

        frame_dir = os.path.join(opts.data_dir, opts.phase+'early', "input", opts.dataset, video)
        # frame_dir = os.path.join(opts.data_dir, opts.phase, "outputatlas")
        print(frame_dir)
        fw_flow_dir = os.path.join(opts.data_dir, opts.phase+'early', "fw_flow", opts.dataset, video)
        if not os.path.isdir(fw_flow_dir):
            os.makedirs(fw_flow_dir)

        fw_occ_dir = os.path.join(opts.data_dir, opts.phase+'early', "fw_occlusion", opts.dataset, video)
        if not os.path.isdir(fw_occ_dir):
            os.makedirs(fw_occ_dir)

        fw_rgb_dir = os.path.join(opts.data_dir, opts.phase+'early', "fw_flow_rgb", opts.dataset, video)
        if not os.path.isdir(fw_rgb_dir):
            os.makedirs(fw_rgb_dir)

        frame_list = glob.glob(os.path.join(frame_dir, "*.jpg"))

        for t in range(1,len(frame_list) ):
            
            print("Compute flow on %s-%s frame %d" %(opts.dataset, opts.phase, t))

            ### load input images 
            img1 = utils.read_img(os.path.join(frame_dir, "%06d.jpg" %(t)))
            img2 = utils.read_img(os.path.join(frame_dir, "%06d.jpg" %(t + 1)))
            
            ### resize image
            size_multiplier = 64
            H_orig = img1.shape[0]
            W_orig = img1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)
            
            img1 = cv2.resize(img1, (W_sc, H_sc))
            img2 = cv2.resize(img2, (W_sc, H_sc))
        
            with torch.no_grad():

                ### convert to tensor
                img1 = utils.img2tensor(img1).to(device)
                img2 = utils.img2tensor(img2).to(device)
        
                ### compute fw flow
                fw_flow = model(img1, img2)

            
                ### compute bw flow
                bw_flow = model(img2, img1)



            ### resize flow
            fw_flow = utils.resize_flow(fw_flow, W_out = W_orig, H_out = H_orig) 
            bw_flow = utils.resize_flow(bw_flow, W_out = W_orig, H_out = H_orig) 
            
            ### compute occlusion
            fw_occ = utils.detect_occlusion(bw_flow, fw_flow)

            ### save flow
            output_flow_filename = os.path.join(fw_flow_dir, "%06d.flo" %t)
            if not os.path.exists(output_flow_filename):
                utils.save_flo(fw_flow, output_flow_filename)
        
            ### save occlusion map
            output_occ_filename = os.path.join(fw_occ_dir, "%06d.png" %t)
            if not os.path.exists(output_occ_filename):
                utils.save_img(fw_occ, output_occ_filename)

            ### save rgb flow
            output_filename = os.path.join(fw_rgb_dir, "%06d.png" %t)
            if not os.path.exists(output_filename):
                flow_rgb = utils.flow_to_rgb(fw_flow)
                utils.save_img(flow_rgb, output_filename)




