import os
import numpy as np
from glob import glob
import src.models.network_filter as net
from src.models.utils import tensor2img, load_image, InputPadder
import argparse
import random
import torch
from tqdm import tqdm 
import cv2
import src.models.utils as utils


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_filter", default="./pretrained_weights/neural_filter.pth",type=str, help="the ckpt of neural filter network")
parser.add_argument("--fps", default=10, type=int, help="frame per second")
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--dataset', type=str, default="ship", help='dataset to test')
parser.add_argument('--phase', type=str, default="test")
parser.add_argument('--data_dir', type=str, default='data', help='path to data folder')
parser.add_argument('--list_dir', type=str, default='lists', help='path to list folder')
# set random seed
seed = 2023
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# process arguments
args = parser.parse_args()
print(args)

# set gpu

if not torch.cuda.is_available():
    raise Exception("No GPU found, run with cpu")
    device = torch.device("cpu")
else:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device("cuda:{}".format(args.gpu))

# Define neural filter model
filter_net = net.UNet(in_channels=6, out_channels=3, init_features=32)

# load ckpt
ckpt = torch.load(args.ckpt_filter)
filter_net.load_state_dict(ckpt)
filter_net.to(device)
filter_net.eval()


### load trained model

list_filename = os.path.join(args.list_dir, "%s_%s.txt" % (args.dataset, args.phase))
with open(list_filename) as f:
    video_list = [line.rstrip() for line in f.readlines()]
for i in range(len(video_list)):
    video = video_list[i]
    style_root = "data/test/output/atlas/{}/stage_1/output".format(video)
    content_root = "data/test/input/{}_marigolddepth".format(video)
    style_names = sorted(glob(style_root + "/*"))
    content_names = sorted(glob(content_root + "/*"))
    assert len(style_names) == len(content_names), "the number of style frames is different from the number of content frames"
    num_frames = len(style_names)
    print("Processing {} frames".format(num_frames))


    process_filter_dir = "data/test/output/filter/{}".format(video)
    os.makedirs(process_filter_dir, exist_ok=True)
    print("neural filter dir:", process_filter_dir)

    for frame_id in tqdm(range(num_frames)):
        ### neural filter net
        frame_content, org_size = load_image(content_names[frame_id], device=device, resize=False)
        frame_style, _ = load_image(style_names[frame_id], size=org_size, device=device, resize=False)
        padder = InputPadder(frame_content.shape)
        frame_content, frame_style = padder.pad(frame_content, frame_style)

        with torch.no_grad():
            frame_pred = filter_net(torch.cat([frame_content, frame_style], dim=1))
        frame_content, frame_style, frame_pred = tensor2img(frame_content), tensor2img(frame_style), tensor2img(frame_pred)
        frame_content = cv2.resize(frame_content, org_size, cv2.INTER_LINEAR)
        frame_style = cv2.resize(frame_style, org_size, cv2.INTER_LINEAR)
        frame_pred = cv2.resize(frame_pred, org_size, cv2.INTER_LINEAR)
        utils.save_img(frame_pred, "{}/{:06d}.png".format(process_filter_dir, frame_id+1))