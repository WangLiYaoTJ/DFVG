import torch
import numpy as np
import argparse
import cv2
import sys
import os
from PIL import Image

device = torch.device("cuda:0")

from .core.utils.utils import InputPadder
from .core.raft import RAFT
from .core.utils import flow_viz

class RAFTWrapper():
    def __init__(self, model_path, max_long_edge=900):
        args = argparse.Namespace()
        args.small = False
        args.mixed_precision = True
        args.model = model_path
        args.max_long_edge = max_long_edge

        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(args.model))

        self.model = self.model.module
        self.model.to(device)
        self.model.eval()

        self.args = args

    def load_image(self, fn):
        img = np.array(Image.open(fn)).astype(np.uint8)
        if img is None:
            print(f'Error reading file: {fn}')
            sys.exit(1)

        im_h = img.shape[0]
        im_w = img.shape[1]
        long_edge = max(im_w, im_h)
        factor = long_edge / self.args.max_long_edge
        if factor > 1:
            new_w = int(im_w // factor)
            new_h = int(im_h // factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        #将图像转换为 PyTorch tensor，并更改维度顺序为 (channels, height, width)，适配模型输入
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img

    def load_image_list(self, image_files):
        images = []
        for imfile in sorted(image_files):
            images.append(self.load_image(imfile))

        #将图像列表堆叠为一个 4D tensor（batch_size, channels, height, width）
        images = torch.stack(images, dim=0)
        images = images.to(device)

        padder = InputPadder(images.shape)
        return padder.pad(images)[0]

    def load_images(self, fn1, fn2):
        """ load and resize to multiple of 64 """
        images = [fn1, fn2]
        images = self.load_image_list(images)
        im1 = images[0, None]
        im2 = images[1, None]
        return im1, im2

    def compute_flow(self, im1, im2):
        padder = InputPadder(im1.shape)
        im1, im2 = padder.pad(im1, im2)
        _, flow12 = self.model(im1, im2, iters=20, test_mode=True)
        flow12 = flow12.detach().to(device)

        return flow12

    def viz(self, flo):

        # map flow to rgb image
        img = flow_viz.flow_to_image(flo)
        return img

