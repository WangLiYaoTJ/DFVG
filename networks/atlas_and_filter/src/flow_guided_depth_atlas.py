import sys
import torch
import torch.optim as optim
import numpy as np
import argparse
import cv2
import glob
from tqdm import tqdm

from src.models.stage_1.implicit_neural_networks import IMLP
from src.models.stage_1.evaluate import evaluate_model_single
from src.models.stage_1.loss_utils import get_gradient_loss_single, get_rigidity_loss, get_optical_flow_loss
from src.models.stage_1.unwrap_utils import get_tuples, pre_train_mapping, load_input_data_single, save_mask_flow

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

import os
import subprocess

import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def main(config, args):
    maximum_number_of_frames = config["maximum_number_of_frames"]

    # read the first frame of vid path and get its resolution
    frames_list = sorted(glob.glob(os.path.join(args.vid_path, "*g")))
    frame_temp = cv2.imread(frames_list[0])
    resx = frame_temp.shape[1]
    resy = frame_temp.shape[0]
    
    if args.down is not None:
        resx = int(resx / args.down)
        resy = int(resy / args.down)
    
    iters_num = config["iters_num"]

    samples = config["samples_batch"]

    evaluate_every = np.int64(config["evaluate_every"])

    load_checkpoint = config["load_checkpoint"]
    checkpoint_path = config["checkpoint_path"]

    data_folder = Path(args.vid_path)#test/input
    data_folder_ori=Path(args.vid_path_ori)
    results_folder_name = "data/test/output/atlas"

    pretrain_mapping1 = config["pretrain_mapping1"]
    pretrain_iter_number = config["pretrain_iter_number"]

    uv_mapping_scale = config["uv_mapping_scale"]

    use_positional_encoding_mapping1 = config["use_positional_encoding_mapping1"]
    number_of_positional_encoding_mapping1 = config["number_of_positional_encoding_mapping1"]
    number_of_layers_mapping1 = config["number_of_layers_mapping1"]
    number_of_channels_mapping1 = config["number_of_channels_mapping1"]

    number_of_channels_atlas = config["number_of_channels_atlas"]
    number_of_layers_atlas = config["number_of_layers_atlas"]
    positional_encoding_num_atlas = config[
        "positional_encoding_num_atlas"]


    rgb_coeff = config["rgb_coeff"]
    optical_flow_coeff = config["optical_flow_coeff"]
    use_gradient_loss = config["use_gradient_loss"]
    gradient_loss_coeff = config["gradient_loss_coeff"]
    rigidity_coeff = config["rigidity_coeff"]
    derivative_amount = config["derivative_amount"]
    include_global_rigidity_loss = config["include_global_rigidity_loss"]
    global_rigidity_derivative_amount_fg = config["global_rigidity_derivative_amount_fg"]
    global_rigidity_coeff_fg = config["global_rigidity_coeff_fg"]
    stop_global_rigidity = config["stop_global_rigidity"]

    times=args.times

    use_optical_flow = True
    vid_name=args.vid_name
    slice_name = data_folder.name
    vid_root = data_folder.parent
    vid_root_ori=data_folder_ori.parent

    results_folder = Path(
        f'{results_folder_name}/{vid_name}/stage_all/{slice_name}/stage_1')

    results_folder.mkdir(parents=True, exist_ok=True)
    with open('%s/config.json' % results_folder, 'w') as json_file:
        json.dump(config, json_file, indent=4)

    writer = SummaryWriter(log_dir=str(results_folder))

    optical_flows_mask, video_frames, optical_flows_reverse_mask, mask_frames, video_frames_dx, video_frames_dy, optical_flows_reverse, optical_flows = load_input_data_single(
        resy, resx, maximum_number_of_frames, data_folder, True,  True, vid_root_ori, slice_name)
    number_of_frames=video_frames.shape[3]
    save_mask_flow(optical_flows_mask, video_frames, results_folder)

    model_F_mapping1 = IMLP(
        input_dim=3,
        output_dim=2,
        hidden_dim=number_of_channels_mapping1,
        use_positional=use_positional_encoding_mapping1,
        positional_dim=number_of_positional_encoding_mapping1,
        num_layers=number_of_layers_mapping1,
        skip_layers=[]).to(device)

    model_F_atlas = IMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=number_of_channels_atlas,
        use_positional=True,
        positional_dim=positional_encoding_num_atlas,
        num_layers=number_of_layers_atlas,
        skip_layers=[4, 7]).to(device)

    start_iteration = 0

    optimizer_all = optim.Adam(
        [{'params': list(model_F_mapping1.parameters())},
         {'params': list(model_F_atlas.parameters())}], lr=0.0001)

    larger_dim = np.maximum(resx, resy)
    if not load_checkpoint:
        if pretrain_mapping1:
            model_F_mapping1 = pre_train_mapping(model_F_mapping1, number_of_frames, uv_mapping_scale, resx=resx, resy=resy,
                                                 larger_dim=larger_dim,device=device, pretrain_iters=pretrain_iter_number)
    else:
        init_file = torch.load(checkpoint_path)
        model_F_atlas.load_state_dict(init_file["F_atlas_state_dict"])
        model_F_mapping1.load_state_dict(init_file["model_F_mapping1_state_dict"])
        optimizer_all.load_state_dict(init_file["optimizer_all_state_dict"])
        start_iteration = init_file["iteration"]

    jif_all = get_tuples(number_of_frames, video_frames)

    # Start training!
    for i in tqdm(range(start_iteration, iters_num)):

        if i > stop_global_rigidity:
            global_rigidity_coeff_fg = 0

        inds_foreground = torch.randint(jif_all.shape[1],
                                        (np.int64(samples * 1.0), 1))

        jif_current = jif_all[:, inds_foreground]

        rgb_current = video_frames[jif_current[1, :], jif_current[0, :], :,
                      jif_current[2, :]].squeeze(1).to(device)

        xyt_current = torch.cat(
            (jif_current[0, :] / (larger_dim / 2) - 1, jif_current[1, :] / (larger_dim / 2) - 1,
             jif_current[2, :] / (number_of_frames / 2.0) - 1),
            dim=1).to(device)

        uv_foreground1 = model_F_mapping1(xyt_current)

        alpha = torch.ones(samples, 1).to(device)

        rgb_output1 = (model_F_atlas(uv_foreground1 * 0.5 + 0.5) + 1.0) * 0.5
        rgb_output_foreground = rgb_output1

        if use_gradient_loss:
            gradient_loss = get_gradient_loss_single(video_frames_dx, video_frames_dy, jif_current,
                                               model_F_mapping1, model_F_atlas,
                                               rgb_output_foreground,device,resx,number_of_frames)
        else:
            gradient_loss = 0.0

        rgb_loss = (torch.norm(rgb_output_foreground - rgb_current, dim=1) ** 2).mean()

        rigidity_loss1 = get_rigidity_loss(
            jif_current,
            derivative_amount,
            larger_dim,
            number_of_frames,
            model_F_mapping1,
            uv_foreground1,device,
            uv_mapping_scale=uv_mapping_scale)

        if include_global_rigidity_loss and i <= stop_global_rigidity:
            global_rigidity_loss1 = get_rigidity_loss(
                jif_current,
                global_rigidity_derivative_amount_fg,
                larger_dim,
                number_of_frames,
                model_F_mapping1,
                uv_foreground1,device,
                uv_mapping_scale=uv_mapping_scale)

        flow_loss1 = get_optical_flow_loss(
            jif_current,
            uv_foreground1,
            optical_flows_reverse,
            optical_flows_reverse_mask,
            larger_dim,
            number_of_frames,
            model_F_mapping1,
            optical_flows,
            optical_flows_mask,
            uv_mapping_scale,device,
            use_alpha=True,
            alpha=alpha)

        if include_global_rigidity_loss and i <= stop_global_rigidity:
            loss = rigidity_coeff * (
                        rigidity_loss1) + global_rigidity_coeff_fg * global_rigidity_loss1 + \
                   rgb_loss * rgb_coeff + optical_flow_coeff * (
                               flow_loss1) + gradient_loss * gradient_loss_coeff
        else:
            loss = rigidity_coeff * (rigidity_loss1) + rgb_loss * rgb_coeff + optical_flow_coeff * (
                        flow_loss1) + gradient_loss * gradient_loss_coeff

        optimizer_all.zero_grad()
        loss.backward()
        optimizer_all.step()

        if i % evaluate_every == 0 and i > start_iteration:
            evaluate_model_single(model_F_atlas,
                                  resx,
                                  resy,
                                  number_of_frames,
                                  model_F_mapping1,
                                  video_frames,
                                  results_folder,
                                  i,
                                  mask_frames,
                                  optimizer_all,
                                  writer,
                                  slice_name,
                                  derivative_amount,
                                  uv_mapping_scale,
                                  optical_flows,
                                  optical_flows_mask,
                                  device,
                                  times)

            rgb_img = video_frames[:, :, :, 0].numpy()
            model_F_atlas.train()
            model_F_mapping1.train()


def group_images(input_folder, output_folder, group_size=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    images.sort()

    for i in range(0, len(images), group_size):
        small_folder = os.path.join(output_folder ,f'{i // group_size + 1}')
        os.makedirs(small_folder, exist_ok=True)

        for img in images[i:i + group_size]:
            src_path = os.path.join(input_folder, img)
            dst_path = os.path.join(small_folder, img)
            shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config_flow_100.json")
    parser.add_argument('--vid_name', type=str)
    parser.add_argument('--phase', type=str, default="test")
    parser.add_argument('--data_dir', type=str, default='data', help='path to data folder')
    parser.add_argument('--down', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    select_gpu = "%d" % args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = select_gpu
    
    config_path = "atlas_and_filter/src/config/%s" % args.config
    vid_path = os.path.join(args.data_dir, args.phase, "input",  args.vid_name+'_marigolddepth')
    vid_path_ori = os.path.join(args.data_dir, args.phase, "input",  args.vid_name)
    vid_path_new = os.path.join(args.data_dir, args.phase, "input",  args.vid_name+'_marigolddepth_sliced')

    vid_path_ori_new=os.path.join(args.data_dir, args.phase, "input",  args.vid_name+'_sliced')
    group_images(vid_path_ori,vid_path_ori_new)
    group_images(vid_path,vid_path_new)


    
    # get flow using current video
    cmd = "python atlas_and_filter/src/preprocess_optical_flow.py --vid-path %s --gpu %s " % (vid_path_ori_new, select_gpu)
    print(cmd)
    subprocess.call(cmd, shell=True)

    times = 0

    for folder_name in sorted(os.listdir(vid_path_ori_new), key=int):
        args.times = times
        args.vid_path = os.path.join(vid_path_new, folder_name)
        args.vid_path_ori=os.path.join(vid_path_ori_new, folder_name)
        with open(config_path) as f:
            main(json.load(f), args)
        times += 1
